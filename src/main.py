from moving_mnist_dataset import MovingMNISTDataset

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch

data = MovingMNISTDataset()
img = data[10]

print(img.shape)

plt.imshow(img[3])
plt.show()
