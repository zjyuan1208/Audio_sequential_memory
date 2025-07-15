import os
import argparse
import json
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.io import savemat

plt.style.use('ggplot')
from src.models import ModernAsymmetricHopfieldNetwork, MultilayertPC, SingleLayertPC
from src.utils import *
from src.get_data import *

# path = 'audio'
path = 'speech'
result_path = os.path.join('./results/', path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

num_path = os.path.join('./results/', path, 'numerical')
if not os.path.exists(num_path):
    os.makedirs(num_path)

fig_path = os.path.join('./results/', path, 'fig')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

model_path = os.path.join('./results/', path, 'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

# add parser as varaible of the main class
parser = argparse.ArgumentParser(description='Sequential memories')

parser.add_argument('--seed', type=int, default=[1], nargs='+',
                    help='seed for model init (default: 1); can be multiple, separated by space')
# parser.add_argument('--latent-size', type=int, default=7600,
parser.add_argument('--latent-size', type=int, default=1600,
                    help='hidden size for training (default: 480)')
parser.add_argument('--input-size', type=int, default=4410,
                    help='input size for training (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate for PC')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--nonlinearity', type=str, default='linear',
                    help='nonlinear function used in the model')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'recall', 'PCA'],
                    help='mode of the script: train or recall (just to save time)')
parser.add_argument('--query', type=str, default='online', choices=['online', 'offline'],
                    help='how you query the recall; online means query with true memory at each time, \
                        offline means query with the predictions')
parser.add_argument('--data-type', type=str, default='continuous', choices=['binary', 'continuous'],
                    help='for movie data from UCF101 this should always be continuous')
parser.add_argument('--noise', type=float, default=0,
                    help='std of noise added to the query')
args = parser.parse_args()


def _extract_latent(model, seq, inf_iters, inf_lr, device):
    seq_len, N = seq.shape
    recall = torch.zeros((seq_len, N)).to(device)
    recall[0] = seq[0].clone().detach()
    prev_z = model.init_hidden(1).to(device)

    # infer the latent of the first image
    x = seq[0].clone().detach()
    model.inference(inf_iters, inf_lr, x, prev_z)
    prev_z = model.z.clone().detach()
    latents = [to_np(prev_z)]

    for k in range(1, seq_len):
        prev_z, _ = model(prev_z)
        latents.append(to_np(prev_z))

    latents = np.concatenate(latents, axis=0)

    # PCA
    pca = PCA(n_components=3)
    transformed_latents = pca.fit_transform(latents)

    return transformed_latents


def _plot_recalls(recall, model_name, args):
    seq_len = recall.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        # ax[j].imshow(to_np(recall[j].reshape((3, 64, 64)).permute(1, 2, 0)))
        # ax[j].imshow(to_np(recall[j].reshape((13, 64)).permute(0, 1)))
        ax[j].imshow(to_np(recall[j].reshape((40, 40))))
        ax[j].axis('off')
        ax[j].set_aspect("auto")
    # plt.tight_layout()
    plt.title(f'recall_{model_name}')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.savefig(fig_path + f'/{model_name}_len{seq_len}_query{args.query}', bbox_inches='tight', dpi=200)


def _plot_memory(x):
    seq_len = x.shape[0]
    fig, ax = plt.subplots(1, seq_len, figsize=(seq_len, 1))
    for j in range(seq_len):
        # ax[j].imshow(to_np(x[j].reshape((3, 64, 64)).permute(1, 2, 0)))
        ax[j].imshow(to_np(x[j].reshape((40, 40))))
        ax[j].axis('off')
        ax[j].set_aspect("auto")

    # Set the title for the entire figure and add padding to avoid cutting it off
    fig.suptitle('Memory', y=1.05)
    plt.subplots_adjust(top=0.85, wspace=0, hspace=0)

    # Display and save the figure
    plt.show()
    fig.savefig(fig_path + f'/memory_len{seq_len}', bbox_inches='tight', dpi=200)


def _plot_PC_loss(loss, seq_len, learn_iters, name):
    # plotting loss for tunning; temporary
    plt.figure()
    plt.plot(loss, label='squared error sum')
    plt.legend()
    plt.savefig(fig_path + f'/{name}_losses_len{seq_len}_iters{learn_iters}')


def main(args):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(device)

    # variables for data and model
    learn_iters = args.epochs
    learn_lr = args.lr
    latent_size = args.latent_size
    input_size = args.input_size * 2
    seed = args.seed
    mode = args.mode
    nonlin = args.nonlinearity

    # inference variables: no need to tune too much
    inf_iters = 100
    inf_lr = 1e-2

    MSEs = []
    seq_len = 20

    print(f'Training variables: seq_len:{seq_len}; seed:{seed}')

    # load data
    # Load data from the dataset
    dataset_folder = '/home/zhyuan/Desktop/seq-memory/data/ESC-50/ESC-50-master/audio'
    # dataset_folder = '/home/zhyuan/Desktop/seq-memory/data/Speech/filtered_subset'

    # Count the total number of .wav files for the progress bar
    total_files = sum(len(files) for _, _, files in os.walk(dataset_folder) if any(f.endswith('.wav') for f in files))
    # total_files = sum(len(files) for _, _, files in os.walk(dataset_folder) if any(f.endswith('.flac') for f in files))

    # Process files with a progress bar
    with tqdm(total=total_files, desc="Processing samples") as pbar:

        for root, _, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith('.wav'):
                # if file.endswith('.flac'):
                    datapath = os.path.join(root, file)
                    file_name = os.path.splitext(file)[0]
                    try:
                        # Process each file with load_audio_mfcc
                        seq = load_audio_mfcc(datapath, sample_num=seq_len).to(device)
                        print(f"Processed: {datapath}")
                    except Exception as e:
                        print(f"Error processing {datapath}: {e}")

                    # seq = load_audio_mfcc('/home/zhyuan/Desktop/seq-memory/data/ESC-50/ESC-50-master/audio', sample_num=seq_len).to(
                    #     device)
                    seq = seq[:int(seq_len * input_size)].reshape(int(seq.size(0) / 2), input_size)  # seq_lenx12288
                    seq = seq[:seq_len, :]  # seq_lenx12288

                    # multilayer PC
                    mpc = MultilayertPC(latent_size, input_size, nonlin=nonlin).to(device)
                    m_optimizer = torch.optim.Adam(mpc.parameters(), lr=learn_lr)

                    # # MCHN
                    # hn = ModernAsymmetricHopfieldNetwork(input_size, sep='softmax', beta=5).to(device)

                    if mode == 'train':
                        # train mPC
                        print('Training multi layer tPC')
                        mPC_losses = train_multilayer_tPC(mpc, m_optimizer, seq, learn_iters, inf_iters, inf_lr, device)
                        torch.save(mpc.state_dict(), os.path.join(model_path, f'mPC_audio_len{seq_len}_seed{seed}_{nonlin}_{file_name}.pt'))
                        # _plot_PC_loss(mPC_losses, seq_len, learn_iters, f"mpc_audio")

                    elif mode == 'recall':
                        # mpc
                        mpc.load_state_dict(
                            torch.load(os.path.join(model_path, f'mPC_audio_len{seq_len}_seed{seed}_{nonlin}_{file_name}.pt'),
                                       map_location=torch.device(device)))

                        mpc.eval()

                        with torch.no_grad():
                            inf_iters = 500
                            m_recalls = multilayer_recall(mpc, seq, inf_iters, inf_lr, args, device)
                            array = m_recalls.detach().cpu().numpy()
                            # Save to a .mat file
                            savemat(f'/home/zhyuan/Desktop/seq-memory/results/{path}/recall_tensor/{file_name}_recall.mat', {'tensor': array})
                            print(f'Recall of {file_name} is done!')
                            # os.remove(os.path.join(model_path, f'mPC_audio_len{seq_len}_seed{seed}_{nonlin}_{file_name}.pt'))
                            # array = seq.detach().cpu().numpy()
                            # # Save to a .mat file
                            # savemat('/home/zhyuan/Desktop/seq-memory/results/tensor_memory.mat', {'tensor': array})
                            # hn_recalls = hn_recall(hn, seq, device, args)
                        del mpc, m_optimizer
                        torch.cuda.empty_cache()
                    pbar.update(1)

                        # if seq_len <= 160:
                        #     _plot_recalls(m_recalls, f"mPC_{nonlin}_audio", args)
                        #     _plot_recalls(hn_recalls, f"HN_audio", args)
                        #     _plot_memory(seq)


if __name__ == "__main__":
    for s in args.seed:
        start_time = time.time()
        args.seed = s
        main(args)
        print(f'Seed complete, total time: {time.time() - start_time}')
