from moving_mnist_dataset import MovingMNISTDataset, pred_transform
from context_encoder import ContextAutoencoder
from callbacks import plot_loss_accuracy, plot_reconstructions, save_epoch_data, save_model, plot_noisy_reconstructions

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
a = np.load("results/noise/epoch_data_0.npz")
b = np.load("results/noise/epoch_data_636.npz")

los = list(a["epoch_loss"])+list(b["epoch_loss"])
acc = list(a["epoch_accuracy"])+list(b["epoch_accuracy"])

np.savez("results/noise/epoch_data.npz", epoch_loss=los, epoch_accuracy=acc)
exit()
'''

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ContextAutoencoder(channels=10).to(DEVICE)
checkpoint = torch.load("weights/random_dropout/model_weight_1201_9.pth")
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
episode = checkpoint['episode']

original_data = MovingMNISTDataset(num_frames=NUM_FRAMES, cache=False)
noisy_data = MovingMNISTDataset(num_frames=NUM_FRAMES, transform=pred_transform, cache=False)

training_data_start = 0
training_data_end = int(len(original_data)*0.9)
test_data_start = int(len(original_data)*0.9)
test_data_end = len(original_data)

with torch.no_grad():
    idx = np.random.randint(test_data_start, test_data_end)
    x_sample = original_data[idx].unsqueeze(0).to(DEVICE)
    x_noisy = noisy_data[idx].unsqueeze(0).to(DEVICE)
    x_pred = model(x_noisy)

    err = torch.nn.functional.mse_loss(x_pred, x_sample)
    accuracy = -err.item()
    plot_noisy_reconstructions(epoch, episode, x_sample.squeeze(0).cpu().numpy(), x_pred.squeeze(0).cpu().numpy(), x_noisy.squeeze(0).cpu().numpy(), 10, prefix="noise_test", save=False)
    plt.show()