from moving_mnist_dataset import MovingMNISTDataset
from context_encoder import ContextAutoencoder
from callbacks import plot_loss_accuracy, plot_reconstructions, save_epoch_data, save_model

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch
import numpy as np

NUM_FRAMES = 1
BATCH_SIZE = 1000
TOTAL_EPOCHS = 2
SAVE_INTERVAL = 10000
PLT_INTERVAL = 10000

'''
epoch_data = np.load("results/epoch_data.npz")
print(epoch_data["epoch_loss"].shape)
print(epoch_data["epoch_accuracy"][0])

model = ContextAutoencoder(channels=10-NUM_FRAMES+1)
checkpoint = torch.load("weights/model_weight_10_90.pth")
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
episode = checkpoint['episode']
'''

original_data = MovingMNISTDataset(num_frames=NUM_FRAMES)

print(len(original_data))
exit()


training_data_start = 0
training_data_end = int(len(original_data)*0.9)
test_data_start = int(len(original_data)*0.9)
test_data_end = len(original_data)


with torch.no_grad():
    x_sample = original_data[np.random.randint(test_data_start, test_data_end-1)].unsqueeze(0)
    x_pred = model(x_sample)

    err = torch.nn.functional.mse_loss(x_pred, x_sample)
    accuracy = -err.item()
    plot_reconstructions(epoch, episode, x_sample.squeeze(0).cpu().numpy(), x_pred.squeeze(0).cpu().numpy(), 10-NUM_FRAMES+1)
    plt.show()
