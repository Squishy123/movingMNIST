from moving_mnist_dataset import MovingMNISTDataset, noisy_transform, default_image_transform
from context_encoder import ContextAutoencoder
from callbacks import plot_loss_accuracy, plot_reconstructions, save_epoch_data, save_model, plot_noisy_reconstructions
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch
import numpy as np

ROOT = Path((Path(__file__).parent / '../').resolve())
RESULTS_PATH = Path(str(ROOT) + "/results/noise")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = Path(str(ROOT) + "/weights/noise")
WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 1
BATCH_SIZE = 1000
TOTAL_EPOCHS = 500
SAVE_INTERVAL = 10000
PLT_INTERVAL = 10000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


original_data = MovingMNISTDataset(num_frames=NUM_FRAMES, transform=default_image_transform)
noisy_data = MovingMNISTDataset(num_frames=NUM_FRAMES, transform=noisy_transform)

# print(original_data[0].shape)
# print(noisy_data[0].shape)

# print(original_data[0:100].shape)
'''
fig, (a, b, c) = plt.subplots(1, 3)
a.imshow(noisy_transform(original_data[0][1].unsqueeze(0)).squeeze(0))
b.imshow(noisy_data[0][2])
c.imshow(original_data[0][2])
plt.show()
exit()
'''

training_data_start = 0
training_data_end = int(len(original_data)*0.9)
test_data_start = int(len(original_data)*0.9)
test_data_end = len(original_data)

model = ContextAutoencoder(channels=10-NUM_FRAMES+1).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
print("BEGINNING TRAINING")
i_count = 0
for epoch in range(TOTAL_EPOCHS):
    print(f"STARTING EPOCH: {epoch+1}")

    epoch_loss = 0
    accuracy = 0
    e_count = 0
    for episode in range((training_data_end-training_data_start)//BATCH_SIZE):
        optim.zero_grad()

        current_state = original_data[episode*BATCH_SIZE:(episode+1)*BATCH_SIZE].to(DEVICE)
        noisy_state = noisy_data[episode*BATCH_SIZE:(episode+1)*BATCH_SIZE].to(DEVICE)
        computed_state = model(current_state)

        predicted_loss = torch.nn.functional.mse_loss(computed_state, current_state)

        predicted_loss.backward()

        for param in model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()

        i_count += BATCH_SIZE
        e_count += BATCH_SIZE
        epoch_loss += predicted_loss.item()

        print(f"EPISODE {episode+1} LOSS: {predicted_loss.item()/BATCH_SIZE}")

        if i_count % PLT_INTERVAL == 0 and i_count != 0:
            with torch.no_grad():
                idx = np.random.randint(test_data_start, test_data_end-1)
                x_sample = original_data[idx].unsqueeze(0).to(DEVICE)
                x_noisy = noisy_data[idx].unsqueeze(0).to(DEVICE)
                x_pred = model(x_noisy)

                err = torch.nn.functional.mse_loss(x_pred, x_sample)
                accuracy = -err.item()
                plot_loss_accuracy(epoch, episode, epoch_loss/e_count, accuracy, prefix="noise")
                plot_noisy_reconstructions(epoch, episode, x_sample.squeeze(0).cpu().numpy(), x_pred.squeeze(0).cpu().numpy(), x_noisy.squeeze(0).cpu().numpy(), 10, prefix="noise")

    # if i_count % SAVE_INTERVAL == 0:
    # SAVE EVERY EPOCH
    save_model(epoch, episode, optim, model, prefix="noise")
    save_epoch_data(epoch, episode, epoch_loss/i_count, accuracy, prefix="noise")
