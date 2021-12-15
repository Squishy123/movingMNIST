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
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noisy = tensor + torch.randn(tensor.size()) * self.std + self.mean
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


noisy_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(32),
    transforms.Normalize((0.1307,), (0.3081,)),
    AddGaussianNoise(0., 1.)
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
"""


class MovingMNISTDataset(Dataset):
    def __init__(self, root=str(DEFAULT_DATASET_LOCATION), transform=default_image_transform, target_transform=None, download=True, frame_skip=2):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.frame_skip = frame_skip

        # Download file
        if not path.exists(self.root + "/mnist_test_seq.npy"):
            if not self.download:
                raise RuntimeError('Dataset not found. Use download=True to download it.')

            res = requests.get(DATASET_NETWORK_URL)
            open(root + "/mnist_test_seq.npy", 'wb').write(res.content)

        # Load file and transpose to pytorch format
        self.data = np.load(self.root + "/mnist_test_seq.npy").transpose(1, 0, 2, 3)[:, ::self.frame_skip]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = torch.tensor(self.data[index]).float()
        img = item

        transform = self.transform
        target_transform = self.target_transform

        # multi select
        if len(item.shape) > 3:
            if transform != None:
                img = np.array([transform(img[i]).numpy() for i in range(len(img))])
                img = torch.tensor(img)

            if target_transform != None:
                target = np.array([transform(target[i]) for i in range(len(target))])
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
