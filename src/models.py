import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.linalg import inv
from src.utils import *
import math
import utils as utils

device = "cuda" if torch.cuda.is_available() else "cpu"


class KalmanFilter(nn.Module):
    """Kalman filter

    x: observation layer
    z: hidden layer
    """
    def __init__(self, A, B, C, Q, R, latent_size) -> None:
        super().__init__()
        self.A = A.clone()
        self.B = B.clone()
        self.C = C.clone()
        # control input, a list/1d array
        self.latent_size = latent_size
        # covariance matrix of noise
        self.Q = Q
        self.R = R
        
    def projection(self):
        z_proj = torch.matmul(self.A, self.z) + torch.matmul(self.B, self.u)
        P_proj = torch.matmul(self.A, torch.matmul(self.P, self.A.t())) + self.Q
        return z_proj, P_proj

    def correction(self, z_proj, P_proj):
        """Correction step in KF

        K: Kalman gain
        """
        K = torch.matmul(torch.matmul(P_proj, self.C.t()), inv(torch.matmul(torch.matmul(self.C, P_proj), self.C.t()) + self.R))
        self.z = z_proj + torch.matmul(K, self.x - torch.matmul(self.C, z_proj))
        self.P = P_proj - torch.matmul(K, torch.matmul(self.C, P_proj))

    def inference(self, inputs, controls):
        zs = []
        pred_xs = []
        exs = []
        seq_len = inputs.shape[1]
        # initialize mean and covariance estimates of the latent state
        self.z = torch.zeros((self.latent_size, 1)).to(inputs.device)
        self.P = torch.eye(self.latent_size).to(inputs.device)
        for l in range(seq_len):
            self.x = inputs[:, l:l+1]
            self.u = controls[:, l:l+1]
            z_proj, P_proj = self.projection()
            self.correction(z_proj, P_proj)
            zs.append(self.z.detach().clone())
            pred_x = torch.matmul(self.C, z_proj)
            pred_xs.append(pred_x)
            exs.append(self.x - pred_x)
        # collect predictions on the observaiton level
        pred_xs = torch.cat(pred_xs, dim=1)
        self.exs = torch.cat(exs, dim=1)
        zs = torch.cat(zs, dim=1)
        return zs, pred_xs

class DiagonalLinear(nn.Module):
    def __init__(self, hidden_size):
        super(DiagonalLinear, self).__init__()
        # Define a learnable parameter for the diagonal entries
        self.diag_weights = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Create the diagonal matrix using diag_weights
        diag_matrix = torch.diag(self.diag_weights)
        # Perform the matrix multiplication
        return x @ diag_matrix  # Equivalent to Wx, where W is diagonal


class FDTDWeightMatrix(nn.Module):
    """
    PyTorch implementation of the FDTD weight matrix that internally simulates
    wave propagation physics but only takes pressure field as input and output.

    This matrix implements a wave equation with damping using the Laplacian operator.
    The implementation internally computes:
    1. Laplacian of the pressure field
    2. Wave propagation with appropriate speed
    3. Damping applied to the pressure field

    Parameters:
    -----------
    grid_side : int
        Side length of the grid (grid_size = grid_side * grid_side)
    dt : float
        Time step size
    dx : float
        Grid spacing
    """

    def __init__(self, grid_side, dt, dx):
        super().__init__()
        self.grid_side = grid_side
        self.dt = dt
        self.dx = dx
        self.grid_size = grid_side * grid_side

        # Initialize trainable parameters
        # Wave speed parameter (one value per grid cell)
        # self.c = nn.Parameter(torch.randn(self.grid_size) * 0.01)

        # Only store the diagonal entries of the damping matrices
        # (grid_side values instead of grid_size)
        self.kp_diag = nn.Parameter(torch.ones(grid_side) * 0.01, requires_grad=True)
        self.k_diag = nn.Parameter(torch.ones(grid_side) * 0.01, requires_grad=True)

        # Instead of random initialization, try a structured initialization
        # that creates gradients of wave speeds across the grid
        indices = torch.arange(self.grid_size, dtype=torch.float)
        x_indices = indices % self.grid_side
        y_indices = indices // self.grid_side

        # Create wave speed gradients based on spatial position
        max_c = 200
        base_speed = 0.3 * max_c  # Use a fraction of the max stable speed
        x_gradient = x_indices / self.grid_side
        y_gradient = y_indices / self.grid_side
        gradient = x_gradient + y_gradient  # Combine gradients

        self.c = nn.Parameter(base_speed * (0.5 + 0.5 * gradient))

    def compute_laplacian(self, p):
        """
        Compute Laplacian of pressure field using finite differences.

        Parameters:
        -----------
        p : torch.Tensor
            Pressure field tensor of shape (batch_size, grid_size)

        Returns:
        --------
        torch.Tensor
            Laplacian of pressure field with shape (batch_size, grid_size)
        """
        batch_size = p.size(0)
        p_2d = p.view(batch_size, self.grid_side, self.grid_side)

        # Compute Laplacian using 5-point stencil
        lap_p = (
                        torch.roll(p_2d, -1, dims=1) +  # up
                        torch.roll(p_2d, 1, dims=1) +  # down
                        torch.roll(p_2d, -1, dims=2) +  # right
                        torch.roll(p_2d, 1, dims=2) -  # left
                        4 * p_2d  # center
                ) / (self.dx ** 2)

        return lap_p.reshape(batch_size, -1)

    def compute_enhanced_laplacian(self, p):
        """
        Compute enhanced Laplacian with additional interaction terms.

        Parameters:
        -----------
        p : torch.Tensor
            Pressure field tensor of shape (batch_size, grid_size)

        Returns:
        --------
        torch.Tensor
            Enhanced Laplacian with additional interactions
        """
        batch_size = p.size(0)
        p_2d = p.view(batch_size, self.grid_side, self.grid_side)

        # Standard 5-point stencil Laplacian
        lap_standard = (
                               torch.roll(p_2d, -1, dims=1) +  # up
                               torch.roll(p_2d, 1, dims=1) +  # down
                               torch.roll(p_2d, -1, dims=2) +  # right
                               torch.roll(p_2d, 1, dims=2) -  # left
                               4 * p_2d  # center
                       ) / (self.dx ** 2)

        # Add diagonal interactions (completing a 9-point stencil)
        lap_diag = (
                           torch.roll(torch.roll(p_2d, -1, dims=1), -1, dims=2) +  # up-right
                           torch.roll(torch.roll(p_2d, -1, dims=1), 1, dims=2) +  # up-left
                           torch.roll(torch.roll(p_2d, 1, dims=1), -1, dims=2) +  # down-right
                           torch.roll(torch.roll(p_2d, 1, dims=1), 1, dims=2)  # down-left
                   ) / (2 * self.dx ** 2)  # Weighted by diagonal distance

        # Add longer-range interactions (2 cells away)
        lap_long = (
                           torch.roll(p_2d, -2, dims=1) +  # 2 up
                           torch.roll(p_2d, 2, dims=1) +  # 2 down
                           torch.roll(p_2d, -2, dims=2) +  # 2 right
                           torch.roll(p_2d, 2, dims=2)  # 2 left
                   ) / (4 * self.dx ** 2)  # Lower weight for longer distance

        # Add frequency-selective coupling (spatial filtering)
        # This simulates how different frequency components interact
        high_freq = p_2d - torch.nn.functional.avg_pool2d(
            p_2d.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1
        ).squeeze(1)

        low_freq = torch.nn.functional.avg_pool2d(
            p_2d.unsqueeze(1),
            kernel_size=5,
            stride=1,
            padding=2
        ).squeeze(1)

        # Coupling between frequency components
        freq_coupling = high_freq * low_freq * 0.1

        # Add directional bias for temporal encoding
        # This creates a preference for wave propagation in certain directions
        dir_bias_x = (torch.roll(p_2d, -1, dims=2) - torch.roll(p_2d, 1, dims=2)) * 0.05
        dir_bias_y = (torch.roll(p_2d, -1, dims=1) - torch.roll(p_2d, 1, dims=1)) * 0.05

        # Combine all interaction terms
        enhanced_lap = lap_standard + 0.5 * lap_diag + 0.25 * lap_long + freq_coupling + dir_bias_x + dir_bias_y

        return enhanced_lap.reshape(batch_size, -1)

    def forward(self, p):
        """
        Apply the FDTD operator to the pressure field.

        Parameters:
        -----------
        p : torch.Tensor
            Pressure field tensor of shape (batch_size, grid_size)

        Returns:
        --------
        torch.Tensor
            Updated pressure field tensor of shape (batch_size, grid_size)
        """
        batch_size = p.size(0)

        # Apply CFL stability condition to wave speed
        max_c = 0.7 * self.dx / (self.dt * math.sqrt(2.0))
        c_stable = torch.clamp(F.softplus(self.c), 0.1, max_c)

        # Apply softplus to ensure positive damping coefficients
        kp_diag_pos = F.softplus(self.kp_diag)
        k_diag_pos = F.softplus(self.k_diag)

        # Compute Laplacian of pressure field
        lap_p = self.compute_enhanced_laplacian(p)

        # Step 1: Wave equation update (approximating the entire p-v system)
        # This uses the wave equation: ∂²p/∂t² = c² ∇²p
        # We discretize to: p_new = 2*p - p_old + c²*dt²*∇²p
        # For our recurrent setting, we use p_exp = p + c²*dt²*∇²p
        p_exp = p + (c_stable ** 2) * self.dt * lap_p

        # Step 2: Apply damping
        # Create damping vector for pressure
        p_damp_inv = torch.ones(self.grid_size, device=p.device)

        # Set diagonal elements - this creates effective damping
        diag_indices = torch.arange(0, self.grid_size, self.grid_side + 1)[:self.grid_side]

        # Combined damping effect from both kp and k
        damping_factor = 1.0 / ((1.0 + self.dt * kp_diag_pos) * (1.0 + self.dt * k_diag_pos))
        p_damp_inv[diag_indices] = damping_factor

        # Apply damping elementwise
        p_new = p_exp * p_damp_inv.unsqueeze(0)

        return p_new


class SimplifiedFDTDWeightMatrix(nn.Module):
    def __init__(self, grid_side, dt, dx):
        super().__init__()
        self.grid_side = grid_side
        self.grid_size = grid_side * grid_side

        self.dt = dt
        self.dx = dx

        # Start with a linear weight matrix that's known to work
        self.linear_weights = nn.Parameter(torch.eye(self.grid_size) * 0.8)

        # Add minimal wave component
        self.wave_influence = nn.Parameter(torch.tensor([0.1]))
        self.c = nn.Parameter(torch.ones(self.grid_size) * 0.3)

    def compute_laplacian(self, p):
        """
        Compute Laplacian of pressure field using finite differences.

        Parameters:
        -----------
        p : torch.Tensor
            Pressure field tensor of shape (batch_size, grid_size)

        Returns:
        --------
        torch.Tensor
            Laplacian of pressure field with shape (batch_size, grid_size)
        """
        batch_size = p.size(0)
        p_2d = p.view(batch_size, self.grid_side, self.grid_side)

        # Compute Laplacian using 5-point stencil
        lap_p = (
                        torch.roll(p_2d, -1, dims=1) +  # up
                        torch.roll(p_2d, 1, dims=1) +  # down
                        torch.roll(p_2d, -1, dims=2) +  # right
                        torch.roll(p_2d, 1, dims=2) -  # left
                        4 * p_2d  # center
                ) / (self.dx ** 2)

        return lap_p.reshape(batch_size, -1)

    def forward(self, p):
        # Get linear prediction (what's known to work)
        linear_pred = p @ self.linear_weights.t()

        # Get wave contribution
        lap_p = self.compute_laplacian(p)
        wave_pred = p + self.c * lap_p

        # Mix with learned ratio
        wave_ratio = torch.sigmoid(self.wave_influence) * 0.3  # Limit influence initially
        output = (1 - wave_ratio) * linear_pred + wave_ratio * wave_pred

        return output

class MultilayertPC(nn.Module):
    """Multi-layer tPC class, using autograd"""
    def __init__(self, hidden_size, output_size, nonlin='tanh'):
        super(MultilayertPC, self).__init__()
        self.hidden_size = hidden_size
        # self.Win = nn.Linear(output_size, hidden_size, bias=False)
        # self.Win.requires_grad_(True)
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.Wr = FDTDWeightMatrix(grid_side=40, dt=3.0, dx=1.0)
        # self.Wr = SimplifiedFDTDWeightMatrix(grid_side=40, dt=3.0, dx=1.0)
        # self.Wr = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        # self.Wr = DiagonalLinear(hidden_size)
        # self.Wr.requires_grad_(True)
        # self.non_linear = nn.Parameter(EulerMatrixGenerator(leaky_rate=1, n=40).generate(delta_k=None))
        # self.non_linear.requires_grad = False
        self.Wout = nn.Linear(hidden_size, output_size, bias=False)
        self.Wout.requires_grad_(True)
        self.dropout = nn.Dropout(p=0.5)

        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")

    def forward(self, prev_z):
        pred_z = self.Wr(self.nonlin(prev_z))
        # pred_z = self.nonlin(self.Win(x)+self.Wr(prev_z))
        pred_x = self.Wout(self.nonlin(pred_z))
        return pred_z, pred_x

    def init_hidden(self, bsz):
        """Initializing prev_z"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))

    def update_errs(self, x, prev_z):
        pred_z, _ = self.forward(prev_z)
        pred_x = self.Wout(self.nonlin(self.z))
        err_z = self.z - pred_z
        err_x = x - pred_x

        # # Compute cosine similarity between current and previous hidden states
        # cosine_similarity = torch.cosine_similarity(prev_z, pred_z, dim=-1) + 1 # Batch-wise similarity
        #
        # # Regularization term: Penalize high similarity
        # regularization_loss = cosine_similarity.mean()  # Average over the batch
        return err_z, err_x

    def update_nodes(self, x, prev_z, inf_lr, update_x=False):
        err_z, err_x = self.update_errs(x, prev_z)

        # dot_product = torch.sum(self.z * prev_z, dim=-1)  # Dot product between pred_z and prev_z
        # norm_pred_z = torch.norm(self.z, p=2, dim=-1)  # Norm of pred_z
        # norm_prev_z = torch.norm(prev_z, p=2, dim=-1)  # Norm of prev_z
        #
        # grad_pred_z = (
        #         prev_z / (norm_pred_z.unsqueeze(-1) * norm_prev_z.unsqueeze(-1) + 1e-8)
        #         - (err_sim.unsqueeze(-1) * self.z) / (norm_pred_z.unsqueeze(-1) ** 2 + 1e-8)
        # )
        #
        # delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone()) + 0.02 * grad_pred_z
        delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
        self.z -= inf_lr * delta_z
        # self.non_linear = nn.Parameter(EulerMatrixGenerator(leaky_rate=1, n=40).generate(delta_k=torch.mean(delta_z, dim=0), lr=0))
        if update_x:
            delta_x = err_x
            x -= inf_lr * delta_x

    def inference(self, inf_iters, inf_lr, x, prev_z, update_x=False):
        """prev_z should be set up outside the inference, from the previous timestep

        Args:
            train: determines whether we are at the training or inference stage

        After every time step, we change prev_z to self.z
        """
        with torch.no_grad():
            # initialize the current hidden state with a forward pass
            self.z, _ = self.forward(prev_z)

            # update the values nodes
            for i in range(inf_iters):
                self.update_nodes(x, prev_z, inf_lr, update_x)

    def update_grads(self, x, prev_z):
        """x: input at a particular timestep in stimulus

        Could add some sparse penalty to weights
        """
        err_z, err_x = self.update_errs(x, prev_z)
        self.hidden_loss = torch.sum(err_z**2)
        self.obs_loss = torch.sum(err_x**2)
        # energy = self.hidden_loss + self.obs_loss + 0.02 * err_sim
        energy = self.hidden_loss + self.obs_loss
        return energy



# class MultilayertPC(nn.Module):
#     """Multi-layer tPC with attention for temporal predictions (replacing Wr)"""
#     def __init__(self, hidden_size, output_size, num_heads=1, nonlin='tanh'):
#         super(MultilayertPC, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#
#         # Replace Wr with attention mechanism
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
#
#         # Wout remains unchanged
#         self.Wout = nn.Linear(hidden_size, output_size, bias=False)
#         self.Wout.requires_grad_(True)
#
#         # Dropout and nonlinearity
#         self.dropout = nn.Dropout(p=0.5)
#
#         if nonlin == 'linear':
#             self.nonlin = Linear()
#         elif nonlin == 'tanh':
#             self.nonlin = Tanh()
#         else:
#             raise ValueError("No such nonlinearity!")
#
#     def forward(self, prev_z):
#         """
#         prev_z: Hidden state from the previous timestep (batch_size, hidden_size)
#         """
#         # Prepare query, key, and value for attention
#         query = prev_z.unsqueeze(1)  # (batch_size, 1, hidden_size)
#         key = prev_z.unsqueeze(1)    # (batch_size, 1, hidden_size)
#         value = prev_z.unsqueeze(1)  # (batch_size, 1, hidden_size)
#
#         # Apply attention mechanism
#         attn_output, _ = self.attention(query, key, value)  # (batch_size, 1, hidden_size)
#         attn_output = attn_output.squeeze(1)               # (batch_size, hidden_size)
#
#         # Nonlinearity and dropout
#         pred_z = self.dropout(self.nonlin(attn_output))
#
#         # Compute the prediction for x
#         pred_x = self.Wout(pred_z)
#
#         return pred_z, pred_x
#
#     def init_hidden(self, bsz):
#         """Initializing prev_z"""
#         return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))
#
#     def update_errs(self, x, prev_z):
#         pred_z, _ = self.forward(prev_z)
#         pred_x = self.Wout(self.nonlin(self.z))
#         err_z = self.z - pred_z
#         err_x = x - pred_x
#         return err_z, err_x
#
#     def update_nodes(self, x, prev_z, inf_lr, update_x=False):
#         err_z, err_x = self.update_errs(x, prev_z)
#         delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
#         self.z -= inf_lr * delta_z
#
#         if update_x:
#             delta_x = err_x
#             x -= inf_lr * delta_x
#
#     def inference(self, inf_iters, inf_lr, x, prev_z, update_x=False):
#         """prev_z should be set up outside the inference, from the previous timestep"""
#         with torch.no_grad():
#             # initialize the current hidden state with a forward pass
#             self.z, _ = self.forward(prev_z)
#
#             # update the values nodes
#             for i in range(inf_iters):
#                 self.update_nodes(x, prev_z, inf_lr, update_x)
#
#     def update_grads(self, x, prev_z):
#         """x: input at a particular timestep in stimulus"""
#         err_z, err_x = self.update_errs(x, prev_z)
#         self.hidden_loss = torch.sum(err_z**2)
#         self.obs_loss = torch.sum(err_x**2)
#         energy = self.hidden_loss + self.obs_loss
#         return energy

# class MultilayertPC(nn.Module):
#     """Multi-layer tPC class, using autograd"""
#
#     def __init__(self, hidden_size, output_size, nonlin='tanh'):
#         super(MultilayertPC, self).__init__()
#         self.hidden_size = hidden_size
#         self.Win = nn.Linear(output_size, hidden_size, bias=False)
#         self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
#         # self.Wr = nn.Parameter(EulerMatrixGenerator(leaky_rate=1, n=20).generate(delta_k=None), requires_grad=False)
#         # self.non_linear = nn.Parameter(EulerMatrixGenerator(leaky_rate=1, n=40).generate(delta_k=None))
#         # self.non_linear.requires_grad = False
#         # self.Wr.requires_grad = False
#         self.Wout = nn.Linear(hidden_size, output_size, bias=False)
#         self.Wout.requires_grad_(True)
#         self.dropout = nn.Dropout(p=0.5)
#
#         if nonlin == 'linear':
#             self.nonlin = Linear()
#         elif nonlin == 'tanh':
#             self.nonlin = Tanh()
#         else:
#             raise ValueError("no such nonlinearity!")
#
#     def forward(self, prev_z, x):
#         # pred_z = self.Wr(self.nonlin(prev_z))
#         # pred_z = self.nonlin(self.Win(x) + self.Wr(prev_z))
#         pred_z = self.nonlin(self.Win(x) + self.Wr @ prev_z.t())
#         pred_x = self.nonlin(self.Wout(pred_z))
#         return pred_z, pred_x
#
#     def init_hidden(self, bsz):
#         """Initializing prev_z"""
#         return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))
#
#     def update_errs(self, x, prev_z):
#         pred_z, _ = self.forward(prev_z, x)
#         pred_x = self.nonlin(self.Wout(self.z))
#         err_z = self.z - pred_z
#         err_x = x - pred_x
#         return err_z, err_x
#
#     def update_nodes(self, x, prev_z, inf_lr, update_x=False):
#         err_z, err_x = self.update_errs(x, prev_z)
#         delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
#         # delta_z = err_z - self.Wout.weight.t() @ self.nonlin.deriv(self.Wout(self.z)) @ err_x
#         self.z -= inf_lr * delta_z
#         # self.non_linear = nn.Parameter(EulerMatrixGenerator(leaky_rate=1, n=40).generate(delta_k=torch.mean(delta_z, dim=0), lr=0))
#         if update_x:
#             delta_x = err_x
#             x -= inf_lr * delta_x
#
#     def inference(self, inf_iters, inf_lr, x, prev_z, update_x=False):
#         """prev_z should be set up outside the inference, from the previous timestep
#
#         Args:
#             train: determines whether we are at the training or inference stage
#
#         After every time step, we change prev_z to self.z
#         """
#         with torch.no_grad():
#             # initialize the current hidden state with a forward pass
#             self.z, _ = self.forward(prev_z, x)
#
#             # update the values nodes
#             for i in range(inf_iters):
#                 self.update_nodes(x, prev_z, inf_lr, update_x)
#
#     def update_grads(self, x, prev_z):
#         """x: input at a particular timestep in stimulus
#
#         Could add some sparse penalty to weights
#         """
#         err_z, err_x = self.update_errs(x, prev_z)
#         self.hidden_loss = torch.sum(err_z ** 2)
#         self.obs_loss = torch.sum(err_x ** 2)
#         energy = self.hidden_loss + self.obs_loss
#         return energy


class SingleLayertPC(nn.Module):
    """Generic single layer tPC"""
    def __init__(self, input_size, nonlin='tanh'):
        super(SingleLayertPC, self).__init__()
        self.Wr = nn.Linear(input_size, input_size, bias=False)
        if nonlin == 'linear':
            self.nonlin = Linear()
        elif nonlin == 'tanh':
            self.nonlin = Tanh()
        else:
            raise ValueError("no such nonlinearity!")
        
        self.input_size = input_size
        
    def init_hidden(self, bsz):
        """Initializing sequence"""
        return nn.init.kaiming_uniform_(torch.empty(bsz, self.input_size))
    
    def forward(self, prev):
        pred = self.Wr(self.nonlin(prev))
        return pred

    def update_errs(self, curr, prev):
        """
        curr: current observation
        prev: previous observation
        """
        pred = self.forward(prev)
        err = curr - pred
        return err
    
    def get_energy(self, curr, prev):
        err = self.update_errs(curr, prev)
        energy = torch.sum(err**2)
        return energy

class LinearSingleLayertPC(nn.Module):
    """
    Linear version of the single layer tPC;

    This is for the convenience of searching Pmax for binary patterns

    Training is performed across the whole sequence,
    rather than step-by-step i.e., the loss is the sum over
    all timesteps.
    """
    def __init__(self, input_size, learn_iters=100, lr=1e-2):
        super(LinearSingleLayertPC, self).__init__()
        self.Wr = nn.Linear(input_size, input_size, bias=False)
        self.input_size = input_size
        self.learn_iters = learn_iters
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(list(self.Wr.parameters()), lr=lr)
    
    def forward(self, s):
        pred = self.Wr(s)
        return pred
    
    def recall(self, s):
        pred = self.Wr(s)
        return pred
    
    def get_loss(self, X):
        """X: shape PxN"""
        pred = self.forward(X[:-1]) # (P-1)xN
        loss = self.criterion(pred, X[1:])
        return loss

    def train(self, X):
        losses = []
        for i in range(self.learn_iters):
            self.optimizer.zero_grad()
            loss = self.get_loss(X)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return losses


class AsymmetricHopfieldNetwork(nn.Module):
    
    def __init__(self, input_size):
        super(AsymmetricHopfieldNetwork, self).__init__()
        self.W = torch.zeros((input_size, input_size))
        
    def forward(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons
        output: PxN matrix
        """
        output = torch.sign(torch.matmul(X, self.W.t()))
        return output
        
    def train(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons

        Asymmetric HN's weight is the auto-covariance matrix of patterns
        """
        P, N = X.shape
        self.W = torch.matmul(X[1:].T, X[:-1]) / N

class ModernAsymmetricHopfieldNetwork(nn.Module):
    """
    MAHN. train() function is simply a placeholder since we don't really train these models
    """
    
    def __init__(self, input_size, sep='linear', beta=1):
        super(ModernAsymmetricHopfieldNetwork, self).__init__()
        self.W = torch.zeros((input_size, input_size))
        self.sep = sep
        self.beta = beta
        
    def forward(self, X, s):
        """
        X: stored memories, shape PxN
        s: query, shape (P-1)xN
        output: (P-1)xN matrix
        """
        _, N = X.shape
        if self.sep == 'exp':
            score = torch.exp(torch.matmul(s, X[:-1].t()))
        elif self.sep == 'softmax':
            score = F.softmax(self.beta * torch.matmul(s, X[:-1].t()), dim=1)
        else:
            score = torch.matmul(s, X[:-1].t()) ** int(self.sep)
        output = torch.matmul(score, X[1:])
        
        return output
        
    def train(self, X):
        """
        X: PxN matrix, where P is seq len, N the number of neurons

        Asymmetric HN's weight is the auto-covariance matrix of patterns
        """
        P, N = X.shape
        self.W = torch.matmul(X[1:].T, X[:-1]) / N

        return -1

class EulerMatrixGenerator(nn.Module):
    def __init__(self, leaky_rate=0.03, n=40):
        super(EulerMatrixGenerator, self).__init__()

        self.leaky_rate = leaky_rate
        self.n = n

        self.CN = 0.1
        self.c0 = 300
        self.crr = 0.8


        self.c = self.c0 * torch.ones([n, n]).to(device)
        self.dc = -250 / n
        for i in range(n):
            self.c[:, i] = self.c[:, i] + self.dc * (i - 1)
        self.c = self.c - self.crr * torch.rand([n, n]).to(device) * self.c

        self.dc = 0

        self.dt = 1.0
        # self.dt = 0.01
        self.dx = self.dt / self.CN * torch.max(torch.max(self.c)) * math.sqrt(2)
        self.dy = self.dx


        self.k = 0.1
        self.k = self.k + 2.0 * torch.rand([n, n])
        self.dk = - 0.1 / n
        # k = k / 100
        self.dk = self.dk / 100

        self.dk = self.dk * 0.0

        self.k_x = nn.Parameter(torch.zeros(n, n))  # Trainable matrix of size n x n
        self.k_y = nn.Parameter(torch.zeros(n, n))  # Trainable matrix of size n x n

        # Example of initializing kp
        self.kp = nn.Parameter(0.0001 * torch.ones(n))

        self.Nx1 = torch.zeros((n, n)).to(device)
        self.Nxprime1 = torch.zeros((n, n)).to(device)
        self.Ny1 = torch.zeros((n, n)).to(device)
        self.Nyprime1 = torch.zeros((n, n)).to(device)
        self.M1 = torch.zeros((n, n)).to(device)
        self.Mprime1 = torch.zeros((n, n)).to(device)
        self.Rx1 = torch.zeros((n, n)).to(device)
        self.Rx2 = torch.zeros((n, n)).to(device)
        self.Sx1 = torch.zeros((n, n)).to(device)
        self.Sx2 = torch.zeros((n, n)).to(device)
        self.Ry1 = torch.zeros((n, n)).to(device)
        self.Sy1 = torch.zeros((n, n)).to(device)

        self.M = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Mprime = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Nx = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Nxprime = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Ny = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Nyprime = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Rx = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Sx = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Ry = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Sy = torch.zeros((self.n * self.n, self.n * self.n)).to(device)
        self.Minv = torch.zeros_like(self.M).to(device)
        self.Nxinv = torch.zeros_like(self.Nx).to(device)
        self.Nyinv = torch.zeros_like(self.Ny).to(device)
        self.A = torch.zeros((3 * self.n * self.n, 3 * self.n * self.n)).to(device)
        self.W = torch.zeros_like(self.A).to(device)


        self.dkx = torch.zeros_like(self.c).to(device)
        self.dky = torch.zeros_like(self.c).to(device)


    def change_diagonal(self, matrix):
        diagonal = torch.diag(matrix)
        reciprocal = 1.0 / diagonal
        matrix.diagonal().copy_(reciprocal)
        return matrix

    def generate(self, delta_k=None, lr=0.001):

        if delta_k is not None:
            self.k_x.data -= lr * delta_k[:self.n ** 2].reshape(self.n, self.n)
            self.k_y.data -= lr * delta_k[:self.n ** 2].reshape(self.n, self.n)
        # self.kp = 0.0001 * torch.ones(self.n).to(device)



        for i in range(self.n):
            self.M1[i, i] = 1 / self.dt - self.kp[i] / 2
            self.Mprime1[i, i] = 1 / self.dt + self.kp[i] / 2

        for i in range(1, self.n + 1):
            # if dk != 0:
            for ii in range(1, self.n + 1):
                self.Nx1[ii - 1, ii - 1] = 1 / self.dt + (self.k_x[ii - 1, i - 1] + self.dkx[ii - 1, i - 1] * (i - 1)) / 2
                self.Nxprime1[ii - 1, ii - 1] = 1 / self.dt - (self.k_x[ii - 1, i - 1] + self.dkx[ii - 1, i - 1] * (i - 1)) / 2
                self.Ny1[ii - 1, ii - 1] = 1 / self.dt + (self.k_y[ii - 1, i - 1] + self.dky[ii - 1, i - 1] * (i - 1)) / 2
                self.Nyprime1[ii - 1, ii - 1] = 1 / self.dt - (self.k_y[ii - 1, i - 1] + self.dky[ii - 1, i - 1] * (i - 1)) / 2

            self.M[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.M1
            self.Mprime[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Mprime1
            self.Nx[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Nx1
            self.Nxprime[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Nxprime1
            self.Ny[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Ny1
            self.Nyprime[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Nyprime1

            # if dc != 0:
            for ii in range(1, self.n + 1):
                self.Rx1[ii - 1, ii - 1] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Rx2[ii - 1, ii - 1] = -(self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Sx1[ii - 1, ii - 1] = -(self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Sx2[ii - 1, ii - 1] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Sy1[ii - 1, ii - 1] = -(self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dy
                if ii > 1:
                    self.Sy1[ii - 1, (ii - 2) % self.n] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dy
                self.Ry1[ii - 1, ii - 1] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dy
                if ii < self.n:
                    self.Ry1[ii - 1, ii % self.n] = -(self.c[ii % self.n, i - 1] + self.dc * (i - 1)) / self.dy
            self.Rx[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Rx1
            self.Sx[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Sx1
            if i < self.n:
                self.Rx[(i - 1) * self.n:i * self.n, (i % self.n) * self.n:((i % self.n) + 1) * self.n] = self.Rx2
                self.Sx[(i % self.n) * self.n:((i % self.n) + 1) * self.n, (i - 1) * self.n:i * self.n] = self.Sx2
            self.Ry[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Ry1
            self.Sy[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Sy1

        self.Minv = torch.inverse(self.M)
        self.Nxinv = torch.inverse(self.Nx)
        self.Nyinv = torch.inverse(self.Ny)

        self.A = torch.vstack([
            torch.hstack([self.Minv @ self.Mprime + self.Minv @ self.Rx @ self.Nxinv @ self.Sx + self.Minv @ self.Ry @ self.Nyinv @ self.Sy,
                          self.Minv @ self.Rx @ self.Nxinv @ self.Nxprime,
                          self.Minv @ self.Ry @ self.Nyinv @ self.Nyprime]),
            torch.hstack(
                [self.Nxinv @ self.Sx, self.Nxinv @ self.Nxprime, torch.zeros((self.n ** 2, self.n ** 2), dtype=torch.float).to(device)]),
            torch.hstack(
                [self.Nyinv @ self.Sy, torch.zeros((self.n ** 2, self.n ** 2), dtype=torch.float).to(device), self.Nyinv @ self.Nyprime])
        ])


        self.W = (self.A - (1 - self.leaky_rate) * torch.eye(self.n ** 2 * 3).to(device)) / self.leaky_rate

        # # Create a dense random matrix
        # random = torch.rand([4800, 4800])
        #
        # # Threshold to zero out some values and make the tensor sparse
        # threshold = 0.05  # This can be adjusted to control sparsity level
        #
        # # Apply threshold
        # sparse_matrix = random * (random > threshold)
        #
        # # Convert to sparse format
        # sparse_tensor = sparse_matrix.to_sparse()

        # return self.A, self.W, self.c, self.k
        return self.A
        # return random


class TemporalPCN(nn.Module):
    """Multi-layer tPC class, using autograd"""

    def __init__(self, options):
        super(TemporalPCN, self).__init__()
        self.Wr = nn.Linear(options.Ng, options.Ng, bias=False)
        self.Win = nn.Linear(options.Nv, options.Ng, bias=False)
        self.Wout = nn.Linear(options.Ng, options.Np, bias=False)

        if options.no_velocity:
            self.Win.weight.data.fill_(0)
            self.Win.weight.requires_grad = False

        self.sparse_z = options.lambda_z
        self.weight_decay = options.weight_decay
        if options.out_activation == 'softmax':
            self.out_activation = utils.Softmax()
        elif options.out_activation == 'tanh':
            self.out_activation = utils.Tanh()
        elif options.out_activation == 'sigmoid':
            self.out_activation = utils.Sigmoid()

        if options.rec_activation == 'tanh':
            self.rec_activation = utils.Tanh()
        elif options.rec_activation == 'relu':
            self.rec_activation = utils.ReLU()
        elif options.rec_activation == 'sigmoid':
            self.rec_activation = utils.Sigmoid()

        self.loss = options.loss

    def set_nodes(self, v, prev_z, p):
        """Set the initial value of the nodes;

        In particular, we initialize the hiddden state with a forward pass.

        Args:
            v: velocity input at a particular timestep in stimulus
            prev_z: previous hidden state
            p: place cell activity at a particular timestep in stimulus
        """
        self.z = self.g(v, prev_z)
        self.x = p.clone()
        self.update_err_nodes(v, prev_z)

    def update_err_nodes(self, v, prev_z):
        self.err_z = self.z - self.g(v, prev_z)
        pred_x = self.decode(self.z)
        if isinstance(self.out_activation, utils.Tanh):
            self.err_x = self.x - pred_x
        elif isinstance(self.out_activation, utils.Softmax):
            self.err_x = self.x / (pred_x + 1e-9)
        else:
            self.err_x = self.x / (pred_x + 1e-9) - (1 - self.x) / (1 - pred_x + 1e-9)

    def g(self, v, prev_z):
        return self.rec_activation(self.Wr(prev_z) + self.Win(v))

    def decode(self, z):
        return self.out_activation(self.Wout(z))

    def inference_step(self, inf_lr, v, prev_z):
        """Take a single inference step"""
        Wout = self.Wout.weight.detach().clone()  # shape [Np, Ng]
        if isinstance(self.out_activation, utils.Softmax):
            delta = self.err_z - (self.out_activation.deriv(self.Wout(self.z)) @ self.err_x.unsqueeze(-1)).squeeze(
                -1) @ Wout
        else:
            delta = self.err_z - (self.out_activation.deriv(self.Wout(self.z)) * self.err_x) @ Wout
        delta += self.sparse_z * torch.sign(self.z)
        self.z = self.z - inf_lr * delta

    def inference(self, inf_iters, inf_lr, v, prev_z, p):
        """Run inference on the hidden state"""
        self.set_nodes(v, prev_z, p)
        for i in range(inf_iters):
            with torch.no_grad():  # ensures self.z won't have grad when we call backward
                self.inference_step(inf_lr, v, prev_z)
            self.update_err_nodes(v, prev_z)

    def get_energy(self):
        """Returns the average (across batches) energy of the model"""
        if self.loss == 'CE':
            obs_loss = F.cross_entropy(self.Wout(self.z), self.x)
        elif self.loss == 'BCE':
            obs_loss = F.binary_cross_entropy_with_logits(self.Wout(self.z), self.x)
        else:
            obs_loss = torch.sum(self.err_x ** 2, -1).mean()
        latent_loss = torch.sum(self.err_z ** 2, -1).mean()
        energy = obs_loss + latent_loss
        energy += self.weight_decay * (torch.mean(self.Wr.weight ** 2))

        return energy, obs_loss

# class TemporalPC(nn.Module):
#     def __init__(self, control_size, hidden_size, output_size, nonlin='tanh'):
#         """A more concise and pytorchy way of implementing tPC
#
#         Suitable for image sequences
#         """
#         super(TemporalPC, self).__init__()
#         self.hidden_size = hidden_size
#         self.Win = nn.Linear(control_size, hidden_size, bias=False)
#         self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.nonlinear = nn.Parameter(EulerMatrixGenerator(leaky_rate=1, n=40).generate(delta_k=None), requires_grad=True)
#         # self.euler_gen = EulerMatrixGenerator(leaky_rate=1, n=40)
#         # self.Wr.weight = nn.Parameter(EulerMatrixGenerator(leaky_rate=1, n=10).generate(n=10))
#         # self.Wr.weight.requires_grad = False
#         self.Wout = nn.Linear(hidden_size, output_size, bias=False)
#
#         if nonlin == 'linear':
#             self.nonlin = Linear()
#         elif nonlin == 'tanh':
#             self.nonlin = Tanh()
#         else:
#             raise ValueError("no such nonlinearity!")
#
#     def forward(self, u, prev_z):
#         # nonlinear_matrix = self.euler_gen.generate(delta_k=None)
#         pred_z = self.Win(self.nonlin(u)) + self.Wr(self.nonlin(prev_z))
#         # pred_z = self.Win(self.nonlin(u)) + self.Wr(prev_z @ self.nonlinear)
#         # pred_z = self.Win(u) + prev_z @ self.nonlinear
#         # pred_z = self.Win(self.nonlin(u)) + prev_z @ self.nonlinear
#         pred_x = self.Wout(self.nonlin(pred_z))
#         return pred_z, pred_x
#
#     def init_hidden(self, bsz):
#         """Initializing prev_z"""
#         return nn.init.kaiming_uniform_(torch.empty(bsz, self.hidden_size))
#
#     def update_errs(self, x, u, prev_z):
#         pred_z, _ = self.forward(u, prev_z)
#         pred_x = self.Wout(self.nonlin(self.z))
#         err_z = self.z - pred_z
#         err_x = x - pred_x
#         return err_z, err_x
#
#     def update_nodes(self, x, u, prev_z, inf_lr, update_x=False):
#         print(prev_z)
#
#         err_z, err_x = self.update_errs(x, u, prev_z)
#         delta_z = err_z - self.nonlin.deriv(self.z) * torch.matmul(err_x, self.Wout.weight.detach().clone())
#         self.z -= inf_lr * delta_z
#         if update_x:
#             delta_x = err_x
#             x -= inf_lr * delta_x
#
#     def inference(self, inf_iters, inf_lr, x, u, prev_z, update_x=False):
#         """prev_z should be set up outside the inference, from the previous timestep
#
#         Args:
#             train: determines whether we are at the training or inference stage
#
#         After every time step, we change prev_z to self.z
#         """
#         with torch.no_grad():
#             # initialize the current hidden state with a forward pass
#             self.z, _ = self.forward(u, prev_z)
#
#             # update the values nodes
#             for i in range(inf_iters):
#                 self.update_nodes(x, u, prev_z, inf_lr, update_x)
#
#     def update_grads(self, x, u, prev_z):
#         """x: input at a particular timestep in stimulus
#
#         Could add some sparse penalty to weights
#         """
#         err_z, err_x = self.update_errs(x, u, prev_z)
#         self.hidden_loss = torch.sum(err_z**2)
#         self.obs_loss = torch.sum(err_x**2)
#         energy = self.hidden_loss + self.obs_loss
#         return energy