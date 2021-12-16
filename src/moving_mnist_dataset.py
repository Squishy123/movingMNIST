# Code is adapted from https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html#MNIST
# Customized for the MovingMNIST Dataset from http://www.cs.toronto.edu/~nitish/unsupervised_video/

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

import numpy as np

from pathlib import Path
from os import path
import requests

default_image_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
    # transforms.Resize(32),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
# https://ai.plainenglish.io/denoising-autoencoder-in-pytorch-on-mnist-dataset-a76b8824e57e


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., noise_factor=0.4):
        self.std = std
        self.mean = mean
        self.noise_factor = noise_factor

    def __call__(self, tensor):
        noisy = tensor + self.noise_factor*torch.randn(tensor.size()) * self.std + self.mean
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomFrameDropout(object):
    def __init__(self, num_frame_drop=5, num_frames=10):
        self.num_frame_drop = num_frame_drop
        self.num_frames = num_frames

    def dropout(self, tensor):
        idx = np.random.randint(0, self.num_frames)
        dropout = tensor[0:idx]
        noise = torch.randn(tensor[idx].shape).unsqueeze(0)
        dropout = torch.cat((dropout, noise, tensor[idx+1:]))
        return dropout

    def __call__(self, tensor):
        # print(tensor.shape)
        if self.num_frame_drop == 0 or tensor.shape[0] <= 1:
            return tensor

        dropout = self.dropout(tensor)
        for i in range(0, self.num_frame_drop):
            dropout = self.dropout(dropout)

        return dropout


class LastDropout(object):
    def __init__(self, dropout_index=5):
        self.dropout_index = dropout_index

    def __call__(self, tensor):
        # print(tensor.shape)
        if self.num_frame_drop == 0 or tensor.shape[0] <= 1:
            return tensor

        dropout = tensor[0:self.dropout_index]
        dropout = torch.cat((dropout, torch.randn(tensor[self.dropout_index:].shape)))

        return dropout


noisy_transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Resize(32),
    transforms.Normalize((0.1307,), (0.3081,)),
    AddGaussianNoise(0., 1., 10),
    RandomFrameDropout(5)
])

pred_transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Resize(32),
    transforms.Normalize((0.1307,), (0.3081,)),
    AddGaussianNoise(0., 1., 10),
    LastDropout()
])

DATASET_NETWORK_URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"

DEFAULT_DATASET_LOCATION = Path((Path(__file__).parent / '../datasets/movingMNIST/').resolve())
DEFAULT_DATASET_LOCATION.mkdir(parents=True, exist_ok=True)

"""
root: String = Path to the directory to store the dataset files
transform: Array = Array of torchvision transforms to apply to the input images
target_transform: Array = Array of torchvision transforms to apply to the target images
download: Boolean = Whether or not to download the dataset
frame_skip: Integer = Number of frames to skip to reduce size
num_frames: Integer = Number of frames to consider for each example
"""


class MovingMNISTDataset(Dataset):
    def __init__(self, root=str(DEFAULT_DATASET_LOCATION), transform=default_image_transform, target_transform=None, download=True, frame_skip=2, num_frames=1, cache=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.cache = cache
        self.cache_built = True

        self.CACHE_LOCATION = Path(self.root + f"/frames_{self.num_frames}")
        self.BATCH_SIZE = 1000

        # Download file
        if not path.exists(self.root + "/mnist_test_seq.npy"):
            if not self.download:
                raise RuntimeError('Dataset not found. Use download=True to download it.')

            print("DOWNLOADING DATASET")
            res = requests.get(DATASET_NETWORK_URL)
            open(root + "/mnist_test_seq.npy", 'wb').write(res.content)

        # Load file and transpose to pytorch format
        self.data = np.load(self.root + "/mnist_test_seq.npy").transpose(1, 0, 2, 3)[:, ::self.frame_skip]

        # Build cache
        if self.cache:
            if not path.exists(self.root + f"/frames_{self.num_frames}"):
                # if True:
                self.cache_built = False
                self.build_cache()
                self.cache_built = True

    def build_cache(self):
        self.CACHE_LOCATION.mkdir(parents=True, exist_ok=True)

        print("BUILDING CACHE")
        for i in range(len(self)//self.BATCH_SIZE):
            print(f"{i+1}/{len(self)//self.BATCH_SIZE}")
            if self.target_transform:
                data, _ = self[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
            else:
                data = self[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]

            cache_state = data[:, :]
            # print(cache_state.shape)
            cache_state = cache_state.unfold(1, self.num_frames, 1).permute((0, 4, 1, 2, 3))
            # print(cache_state.shape)
            cache_state = cache_state.reshape(cache_state.shape[0] * cache_state.shape[1], cache_state.shape[2], cache_state.shape[3], cache_state.shape[4])
            # print(cache_state.shape)

            np.savez(str(self.CACHE_LOCATION) + f"/CACHE_{i}.npz", state=cache_state)

    def __len__(self):
        if not self.cache_built or not self.cache:
            return len(self.data)
        num_examples = self.data.shape[1]-self.num_frames+1
        total_examples = self.data.shape[0] * num_examples
        return total_examples

    def __getitem__(self, index):
        if self.cache and self.cache_built:
            if type(index) is slice:
                istart = index.start
                istop = index.stop
                if index.start == None:
                    istart = 0
                if index.stop == None:
                    istop = istart + self.BATCH_SIZE

                # print(istart)
                # print(istop)

                start = istart // self.BATCH_SIZE
                stop = istop // self.BATCH_SIZE

                # print(start)
                # print(stop)

                combined_data = None
                for i in range(start, stop+1):
                    data = np.load(str(self.CACHE_LOCATION) + f"/CACHE_{i // self.BATCH_SIZE}.npz")["state"]
                    # print(data.shape)
                    item = torch.tensor(data).float()

                    if combined_data == None:
                        combined_data = item
                    else:
                        combined_data = torch.cat((combined_data, item), 0)
                #item = combined_data[(istart % self.BATCH_SIZE)+start*self.BATCH_SIZE:(istop % self.BATCH_SIZE)+stop*self.BATCH_SIZE]
                item = combined_data[istart % self.BATCH_SIZE: istart % self.BATCH_SIZE + istop-istart]
                # print("VALUES")
                # print((istart % self.BATCH_SIZE)+start*self.BATCH_SIZE)
                # print((istop % self.BATCH_SIZE)+stop*self.BATCH_SIZE)
                # print(combined_data.shape)
            else:
                num_examples = self.data.shape[1]-self.num_frames+1
                data = np.load(str(self.CACHE_LOCATION) + f"/CACHE_{index // self.data.shape[0]}.npz")["state"]
                item = torch.tensor(data[index % self.BATCH_SIZE]).float()
        else:
            item = torch.tensor(self.data[index]).float()

        img = item
        target = item

        transform = self.transform
        target_transform = self.target_transform

        # multi select
        if len(item.shape) > 3:
            if transform != None:
                img = np.array([transform(img[i]).numpy() for i in range(len(img))])
                img = torch.tensor(img)

            if target_transform != None:
                target = np.array([transform(target[i]).numpy() for i in range(len(target))])
                target = torch.tensor(target)
            else:
                return img

            return img, target

        if transform != None:
            img = transform(item)

        if target_transform != None:
            target = target_transform(item)
        else:
            return img

        return img, target
