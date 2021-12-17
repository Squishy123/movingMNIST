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

PREFIX = "pred_dropout"

ROOT = Path((Path(__file__).parent / '../').resolve())
PLOTS_PATH = Path(str(ROOT) + f"/plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
Path(str(PLOTS_PATH)+f"/{PREFIX}").mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ContextAutoencoder(channels=10).to(DEVICE)
checkpoint = torch.load(f"weights/{PREFIX}/model_weight_1_9.pth")
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
episode = checkpoint['episode']

original_data = MovingMNISTDataset(num_frames=NUM_FRAMES, cache=False)
noisy_data = MovingMNISTDataset(num_frames=NUM_FRAMES, transform=pred_transform, cache=False)

training_data_start = 0
training_data_end = int(len(original_data)*0.9)
test_data_start = int(len(original_data)*0.9)
test_data_end = len(original_data)

epoch_data = np.load(f"results/{PREFIX}/epoch_data.npz")

# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

loss_data = list(reject_outliers(epoch_data["epoch_loss"]))
accuracy_data = list(reject_outliers(epoch_data["epoch_accuracy"]))

loss_fig, (loss_ax) = plt.subplots(1,1)
loss_ax.set_title("Model Training Loss")

acc_fig, (acc_ax) = plt.subplots(1,1)
acc_ax.set_title("Model Training Accuracy")

for i in range(len(loss_data)):
    loss_ax.scatter(i, loss_data[i], color="red")

for i in range(len(accuracy_data)):
    acc_ax.scatter(i, accuracy_data[i], color="blue")

loss_fig.savefig(str(PLOTS_PATH) + f"/{PREFIX}/training_loss.png")
acc_fig.savefig(str(PLOTS_PATH) + f"/{PREFIX}/accuracy_loss.png")