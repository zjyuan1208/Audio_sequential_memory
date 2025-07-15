from __future__ import print_function
import os
import os.path
import errno
import torch
from torchvision import datasets, transforms
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import numpy as np
import math
import cv2
import torchaudio
import torchaudio.transforms as T

from src.utils import *
import librosa

class DataWrapper(Dataset):
    """
    Class to wrap a dataset. Assumes X and y are already
    torch tensors and have the right data type and shape.
    
    Parameters
    ----------
    X : torch.Tensor
        Features tensor.
    y : torch.Tensor
        Labels tensor.
    """
    def __init__(self, X):
        self.features = X
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], []
    
def generate_correlated_binary_patterns(P, N, b, device, seed=1):
    np.random.seed(seed)
    X = np.zeros((int(P), int(N)))
    template = np.random.choice([-1, 1], size=N)
    prob = (1 + b) / 2
    for i in range(P):
        for j in range(N):
            if np.random.binomial(1, prob) == 1:
                X[i, j] = template[j]
            else:
                X[i, j] = -template[j]
            
        # revert the sign
        if np.random.binomial(1, 0.5) == 1:
            X[i, j] *= -1

    return to_torch(X, device)

def load_aliased_mnist(seed):
     # Set random seed for PyTorch random number generator
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Set up MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Filter dataset to include only digits 1, 2, and 3
    mnist_123 = [img for img, label in mnist if label in [1, 2, 3]]

    # Sample 5 random indices from the filtered dataset
    indices = torch.randperm(len(mnist_123))[:5]

    # Extract images corresponding to the sampled indices
    sequence = [mnist_123[i] for i in indices]

    # Replace the last two images with the first two images
    sequence[3], sequence[4] = sequence[1], sequence[0]

    # Convert images to PyTorch tensors and stack into a sequence tensor
    sequence_tensor = torch.stack(sequence).squeeze()

    return sequence_tensor

def load_sequence_mnist(seed, seq_len, order=True, binary=True):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 28, 28))

    if order:
        # Loop through each digit class and randomly sample one image from each class
        for i in range(seq_len):
            indices = torch.where(mnist.targets == i)[0]
            idx = torch.randint(0, indices.size()[0], (1,))
            img, _ = mnist[indices[idx][0]]
            sequence[i] = img.squeeze()

    else:
        # Sample `seq_len` random images from the MNIST dataset
        indices = torch.randint(0, len(mnist), (seq_len,))
        for i, idx in enumerate(indices):
            img, _ = mnist[idx]
            sequence[i] = img.squeeze()

    if binary:
        sequence[sequence > 0.5] = 1
        sequence[sequence <= 0.5] = -1

    return sequence

def replace_images(seq, seed, p):
    X = seq.clone()
    N = math.ceil(p * X.shape[0])

    # set random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])
    
    # load mnist test set
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # randomly select N indices from the sequence
    indices = torch.randperm(X.shape[0])[:N]

    # use the same seed for selecting the filling test data
    torch.manual_seed(2023)
    random_index = torch.randint(len(test_set), size=(1,))
    
    # replace images at selected indices with random images from test set
    for i in indices:
        X[i] = test_set[random_index[0]][0].squeeze()
    
    # output the changed sequence
    return X

def load_sequence_emnist(seed, seq_len):
    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    emnist = datasets.EMNIST(root='./data', train=True, split='balanced', download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 28, 28))

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    i = 0
    while i < seq_len:
        idx = torch.randint(len(emnist), (1,))
        image, target = emnist[idx[0]]
        if target >= 10:  # Ignore digits
            sequence[i] = image.squeeze()
            i += 1

    # Sample `seq_len` random images from the MNIST dataset
    # indices = torch.randint(0, len(emnist), (seq_len,))
    # for i, idx in enumerate(indices):
    #     img, _ = emnist[idx]
    #     sequence[i] = img.squeeze()

    return sequence

def load_sequence_cifar(seed, seq_len):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Define the transform to convert the images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the CIFAR10 dataset
    cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Initialize an empty tensor to store the sequence of digits
    sequence = torch.zeros((seq_len, 3, 32, 32))

    # Sample `seq_len` random images from the MNIST dataset
    indices = torch.randint(0, len(cifar), (seq_len,))
    for i, idx in enumerate(indices):
        img, _ = cifar[idx]
        sequence[i] = img

    return sequence

def get_seq_mnist(datapath, seq_len, sample_size, batch_size, seed, device):
    """Get batches of sequence mnist
    
    The data should be of shape [sample_size, seq_len, h, w]
    """
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    # test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # each sample is a sequence of randomly sampled mnist digits
    # we could thus sample samplesize x seq_len images
    random.seed(seed)
    train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size * seq_len))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size * seq_len, shuffle=False)

    return train_loader


def get_mnist(datapath, sample_size, sample_size_test, batch_size, seed, device, binary=False, classes=None):
    # classes: a list of specific class to sample from
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)
    test = datasets.MNIST(datapath, train=False, transform=transform, download=True)

    # subsetting data based on sample size and number of classes
    idx = sum(train.targets == c for c in classes).bool() if classes else range(len(train))
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    if sample_size != len(train):
        random.seed(seed)
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), sample_size))
    random.seed(seed)
    test = torch.utils.data.Subset(test, random.sample(range(len(test)), sample_size_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = [], []
    for batch_idx, (data, targ) in enumerate(train_loader):
        X.append(data)
        y.append(targ)
    X = torch.cat(X, dim=0).to(device) # size, 28, 28
    y = torch.cat(y, dim=0).to(device)

    X_test, y_test = [], []
    for batch_idx, (data, targ) in enumerate(test_loader):
        X_test.append(data)
        y_test.append(targ)
    X_test = torch.cat(X_test, dim=0).to(device) # size, 28, 28
    y_test = torch.cat(y_test, dim=0).to(device)

    if binary:
        X[X > 0.5] = 1
        X[X < 0.5] = 0
        X_test[X_test > 0.5] = 1
        X_test[X_test < 0.5] = 0

    print(X.shape)
    return (X, y), (X_test, y_test)


def get_rotating_mnist(datapath, seq_len, sample_size, batch_size, seed, angle):
    """digit: digit used to train the model
    
    test_digit: digit used to test the generalization of the model

    angle: rotating angle at each step
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.MNIST(datapath, train=True, transform=transform, download=True)

    # randomly sample 
    dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # get data from particular classes
    # idx = (train.targets != test_digit).bool()
    # test_idx = (train.targets == test_digit).bool()
    train_data = train.data / 255.
    # test_data = train.data[test_idx] / 255.

    random.seed(seed)
    train_data = train_data[random.sample(range(len(train_data)), sample_size)] # [sample_size, h, w]
    # test_data = test_data[random.sample(range(len(test_data)), test_size)]
    h, w = train_data.shape[-2], train_data.shape[-1]
    # rotate images
    train_sequences = torch.zeros((sample_size, seq_len, h, w))

    for l in range(seq_len):
        train_sequences[:, l] = TF.rotate(train_data, angle * l)

    train_loader = DataLoader(DataWrapper(train_sequences), batch_size=batch_size)
    
    return train_loader

def read_audio(audio_path, resample=True):
    r"""Loads audio file or array and returns a torch tensor"""
    # Randomly sample a segment of audio_duration from the clip or pad to match duration
    audio_time_series, sample_rate = torchaudio.load(audio_path)
    sampling_rate = 16000
    resample_rate = sampling_rate
    if resample and resample_rate != sample_rate:
        resampler = T.Resample(sample_rate, resample_rate)
        audio_time_series = resampler(audio_time_series)
    return audio_time_series, resample_rate

def load_audio_into_tensor(audio_path, audio_duration, resample=False):
    r"""Loads audio file and returns raw audio."""
    # Randomly sample a segment of audio_duration from the clip or pad to match duration
    audio_time_series, sample_rate = read_audio(audio_path, resample=resample)
    audio_time_series = audio_time_series.reshape(-1)

    # audio_time_series is shorter than predefined audio duration,
    # so audio_time_series is extended
    if audio_duration*sample_rate >= audio_time_series.shape[0]:
        repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                    audio_time_series.shape[0]))
        # Repeat audio_time_series by repeat_factor to match audio_duration
        audio_time_series = audio_time_series.repeat(repeat_factor)
        # remove excess part of audio_time_series
        audio_time_series = audio_time_series[0:audio_duration*sample_rate]
    else:
        # audio_time_series is longer than predefined audio duration,
        # so audio_time_series is trimmed
        start_index = random.randrange(
            audio_time_series.shape[0] - audio_duration*sample_rate)
        audio_time_series = audio_time_series[start_index:start_index +
                                              int(audio_duration*sample_rate)]
    return torch.FloatTensor(audio_time_series)

def load_audio_mfcc(datapath, sample_num):
    # datapath = '/home/zhyuan/Desktop/seq-memory/data/ESC-50/ESC-50-master/audio'
    # audio_files = ['/home/zhyuan/Desktop/seq-memory/data/ESC-50/ESC-50-master/audio/1-103999-A-30.wav']
    # audio_files = ['/home/zhyuan/Desktop/seq-memory/data/Speech/LibriSpeech/dev-clean/1993/147964/1993-147964-0002.flac']
    # audio_files = ['/home/zhyuan/Desktop/seq-memory/data/Speech/LibriSpeech/dev-clean/3853/163249/3853-163249-0005.flac']
    # audio_files = ['/home/zhyuan/Desktop/seq-memory/data/Speech/filtered_subset/8297-275154-0008.flac']
    audio_files = [f'{datapath}']
    duration = 20  # duration in seconds
    chunk_duration = 0.1  # 100 msec
    audio_tensors = []

    for audio_file in audio_files:
        # Load the entire audio file into a tensor with specified duration and resampling if needed
        audio_tensor = load_audio_into_tensor(audio_file, duration, resample=True)

        # Reshape to a single batch and move to GPU if available
        audio_tensor = audio_tensor.reshape(1, -1).cuda() if torch.cuda.is_available() else audio_tensor.reshape(1, -1)

        # Determine the number of samples in each 10-msec chunk
        sample_rate = 44100  # Replace with the actual sample rate used in `load_audio_into_tensor`
        samples_per_chunk = int(chunk_duration * sample_rate)

        # Split the tensor into 10-msec chunks
        chunks = [audio_tensor[:, i:i + samples_per_chunk] for i in range(0, audio_tensor.size(1), samples_per_chunk)]

        # Append list of chunks (for each file) to the audio_tensors list
        audio_tensors.append(chunks)

    audio_tensors = torch.stack(audio_tensors[0][:-1]).squeeze()
    return audio_tensors

# # Define the function to compute MFCCs
# def load_audio_mfcc(datapath, sample_num):
#     audio_files = ['/home/zhyuan/Desktop/seq-memory/data/Speech/filtered_subset/8297-275154-0008.flac']
#     # audio_files = [f'{datapath}']
#     duration = 20  # duration in seconds
#     chunk_duration = 0.1  # 100 msec
#     audio_tensors = []
#
#     for audio_file in audio_files:
#         # Load the entire audio file into a tensor with specified duration and resampling if needed
#         waveform, sample_rate = torchaudio.load(audio_file)
#
#         # Resample the waveform to 44100 Hz if needed
#         target_sample_rate = 44100
#         if sample_rate != target_sample_rate:
#             resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
#             waveform = resampler(waveform)
#             sample_rate = target_sample_rate
#
#         # Truncate or pad the waveform to the specified duration
#         max_samples = int(duration * sample_rate)
#         if waveform.size(1) > max_samples:
#             waveform = waveform[:, :max_samples]
#         else:
#             padding = max_samples - waveform.size(1)
#             waveform = torch.nn.functional.pad(waveform, (0, padding))
#
#         # Determine the number of samples in each 100-msec chunk
#         samples_per_chunk = int(chunk_duration * sample_rate)
#
#         # Split the tensor into 100-msec chunks
#         chunks = [waveform[:, i:i + samples_per_chunk] for i in range(0, waveform.size(1), samples_per_chunk)]
#
#         # Compute MFCCs for each chunk
#         mfcc_transform = torchaudio.transforms.MFCC(
#             sample_rate=sample_rate,
#             n_mfcc=13,  # Number of MFCC coefficients to compute
#             melkwargs={
#                 'n_fft': 400,  # Number of FFT bins
#                 'hop_length': 16,  # Hop length
#                 'n_mels': 40,  # Number of Mel filter banks
#                 'center': False
#             }
#         )
#         mfcc_chunks = [mfcc_transform(chunk).squeeze(0) for chunk in chunks if chunk.size(1) == samples_per_chunk]
#
#         # Append MFCC chunks for the file
#         audio_tensors.append(torch.stack(mfcc_chunks))
#
#     # Combine tensors from all files (only the first file is considered in this example)
#     audio_tensors = torch.cat(audio_tensors, dim=0)
#     return audio_tensors


def load_audio_stft(datapath, sample_num):
    # Parameters
    duration = 5  # Max duration in seconds
    chunk_duration = 0.1  # Duration of each chunk in seconds (100 ms)
    n_fft = 4096  # Number of FFT points (frequency resolution)
    hop_length = n_fft // 4  # Hop length for STFT (75% overlap)
    window = 'hann'  # Window function

    # Initialize an empty list to hold the STFT tensors
    stft_tensors = []

    # Load the audio file into a tensor
    audio_tensor = load_audio_into_tensor(datapath, duration, resample=True)
    audio_tensor = audio_tensor.reshape(1, -1).cuda() if torch.cuda.is_available() else audio_tensor.reshape(1, -1)

    # Determine sample rate and number of samples per chunk
    sample_rate = 16000  # Replace with actual sample rate from `load_audio_into_tensor`
    samples_per_chunk = int(chunk_duration * sample_rate)

    # Split the tensor into 100 ms chunks and compute STFT for each chunk
    for i in range(0, audio_tensor.size(1), samples_per_chunk):
        # Extract a chunk
        chunk = audio_tensor[:, i:i + samples_per_chunk]

        # If chunk is shorter than expected (last chunk), zero-pad
        if chunk.size(1) < samples_per_chunk:
            padding = samples_per_chunk - chunk.size(1)
            chunk = torch.nn.functional.pad(chunk, (0, padding))

        # Convert the chunk to a NumPy array for librosa processing
        chunk_np = chunk.cpu().numpy().squeeze()

        # Compute the STFT using librosa
        stft = librosa.stft(chunk_np, n_fft=n_fft, hop_length=hop_length, window=window)

        # Convert the STFT result to magnitude (real values) and normalize
        stft_magnitude = np.abs(stft)

        # Convert the result back to a PyTorch tensor
        stft_tensor = torch.tensor(stft_magnitude, dtype=torch.float32)

        # Append the STFT tensor to the list
        stft_tensors.append(stft_tensor)

    # Stack all STFT tensors into a single 3D tensor
    stft_tensors = torch.stack(stft_tensors)

    # Return the final tensor
    return stft_tensors.view(stft_tensors.size(0), -1)

def load_ucf_frames(datapath):
    # load UCF101 movies directly from folder

    # Set desired output tensor dimensions
    num_frames, height, width = 10, 64, 64

    # Define data transformation pipeline to resize and convert images to tensors
    data_transforms = transforms.Compose([
        transforms.Resize((height, width)),
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Initialize an empty tensor to store the image tensors
    image_tensors = torch.empty(num_frames, 3, height, width)

    # avi_file_path = datapath
    datapath = '/home/zhyuan/Desktop/seq-memory/data/frame'
    avi_file_path = '/home/zhyuan/Desktop/seq-memory/data/UCF-101/CliffDiving/v_CliffDiving_g01_c01.avi'
    output_folder = '/home/zhyuan/Desktop/seq-memory/data/frame'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(avi_file_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {avi_file_path}")
    else:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read frame {frame_number}")
                break

            # Save the current frame as a JPEG image
            frame_path = os.path.join(output_folder, f'frame_{frame_number:02d}.jpg')
            cv2.imwrite(frame_path, frame)

            frame_number += 1

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

        print(f"Extracted {frame_number} frames from {avi_file_path}")

    # Loop through the JPEG images in the directory and convert them to PyTorch tensors
    for i in range(num_frames):
        # Set the path to the JPEG image file
        image_file = os.path.join(datapath, f'frame_{i:02d}.jpg')
        
        # Load the JPEG image as a PIL Image object
        image = Image.open(image_file)
        
        # Apply the data transformation pipeline to the image and convert it to a tensor
        image_tensor = data_transforms(image)
        
        # Store the tensor in the output tensor
        image_tensors[i] = image_tensor

    return image_tensors



class MovingMNIST(Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([self.transform(img), new_data], dim=0)
            return new_data

        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
def get_moving_mnist(datapath, sample_size, batch_size, seed):
    """
    Load the moving MNIST dataset
    """
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_set = MovingMNIST(root=datapath, train=True, download=True, transform=data_transforms)

    random.seed(seed)
    train_set = torch.utils.data.Subset(train_set, random.sample(range(len(train_set)), sample_size))

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)

    return train_loader

