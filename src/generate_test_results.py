from pathlib import Path
from moving_mnist_dataset import MovingMNISTDataset, pred_transform
from context_encoder import ContextAutoencoder

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch
import numpy as np

NUM_FRAMES = 1
BATCH_SIZE = 1000
TOTAL_EPOCHS = 2
SAVE_INTERVAL = 10000
PLT_INTERVAL = 10000

PREFIX = "default"

ROOT = Path((Path(__file__).parent / '../').resolve())
TESTS_PATH = Path(str(ROOT) + f"/tests/{PREFIX}")
TESTS_PATH.mkdir(parents=True, exist_ok=True)

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

accuracy_tot = []

def plot_test_reconstructions(idx, actual, predicted, noisy, NUM_FRAMES=10):
    recon_fig, ax = plt.subplots(3, NUM_FRAMES, sharex='col', sharey='row')

    ax[0][NUM_FRAMES//2].set_title(f"Test Reconstruction")

    for i in range(NUM_FRAMES):
        ax[0][i].imshow(actual[i])
        ax[1][i].imshow(noisy[i])
        ax[2][i].imshow(predicted[i])
    
    recon_fig.savefig(str(TESTS_PATH) + f"/reconstruction_{idx}.png")

with torch.no_grad():
    print("RUNNING TESTS")
    for i in range(test_data_start, test_data_end, BATCH_SIZE):
        print(f"{i}/{test_data_end}")
        x_sample = original_data[i:i+BATCH_SIZE].to(DEVICE)
        x_noisy = noisy_data[i:i+BATCH_SIZE].to(DEVICE)
        x_pred = model(x_noisy)

        err = torch.nn.functional.mse_loss(x_pred, x_sample)
        vol = x_sample.shape[0]*x_sample.shape[1]*x_sample.shape[2]*x_sample.shape[3]
        accuracy = 1 - err.item()/vol
        accuracy_tot.append(accuracy)

        for j in range(i, i+BATCH_SIZE,100):
            plot_test_reconstructions(j, x_sample[j-i].squeeze(0).cpu().numpy(), x_pred[j-i].squeeze(0).cpu().numpy(), x_noisy[j-i].squeeze(0).cpu().numpy(), 10)

