from moving_mnist_dataset import MovingMNISTDataset

import matplotlib.pyplot as plt

data = MovingMNISTDataset()
img = data[10]

plt.imshow(img[3])
plt.show()
print(img.shape)
