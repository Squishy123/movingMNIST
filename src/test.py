from moving_mnist_dataset import MovingMNISTDataset, pred_transform
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

model = ContextAutoencoder(channels=10)
checkpoint = torch.load("weights/noise_0/model_weight_132_90.pth")
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
episode = checkpoint['episode']

original_data = MovingMNISTDataset(num_frames=NUM_FRAMES, cache=False)
noisy_data = MovingMNISTDataset(num_frames=NUM_FRAMES, transform=pred_transform, cache=False)

print(len(original_data))
exit()


training_data_start = 0
training_data_end = int(len(original_data)*0.9)
test_data_start = int(len(original_data)*0.9)
test_data_end = len(original_data)

with torch.no_grad():
    idx = np.random.randint(10)
    x_sample = original_data[idx].unsqueeze(0).to(DEVICE)
    x_noisy = noisy_data[idx].unsqueeze(0).to(DEVICE)
    x_pred = model(x_noisy)

    err = torch.nn.functional.mse_loss(x_pred, x_sample)
    accuracy = -err.item()
    plot_loss_accuracy(epoch, episode, epoch_loss/e_count, accuracy, prefix="noise_test")
    plot_noisy_reconstructions(epoch, episode, x_sample.squeeze(0).cpu().numpy(), x_pred.squeeze(0).cpu().numpy(), x_noisy.squeeze(0).cpu().numpy(), 10, prefix="noise_test")
