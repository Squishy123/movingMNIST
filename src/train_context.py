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
BATCH_SIZE = 1000 # samples
TOTAL_EPOCHS = 5000
SAVE_INTERVAL = 100 # in epochs
PLT_INTERVAL = 5000 # in samples

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class FrameDropout(object):
    def __init__(self, num_frame_drop=5):
        self.num_frame_drop = num_frame_drop

    def dropout(self, tensor):
        idx = np.random.randint(0, 10/NUM_FRAMES)
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
        '''
        # generate dropout indices
        dropout_indices = []
        for i in range(self.num_frame_drop):
            num = np.random.randint(0, 10/NUM_FRAMES)
            while not num in dropout_indices:
                num = np.random.randint(0, 10/NUM_FRAMES)
            dropout_indices.append(num)
        '''
        return dropout


noisy_transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Resize(32),
    transforms.Normalize((0.1307,), (0.3081,)),
    AddGaussianNoise(0., 1.),
    FrameDropout(5)
])


original_data = MovingMNISTDataset(num_frames=NUM_FRAMES, transform=default_image_transform, cache=False)
noisy_data = MovingMNISTDataset(num_frames=NUM_FRAMES, transform=noisy_transform, cache=False)

print(len(original_data))
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
        computed_state = model(noisy_state)

        predicted_loss = torch.nn.functional.mse_loss(computed_state, current_state)

        predicted_loss.backward()

        for param in model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()

        i_count += BATCH_SIZE
        e_count += BATCH_SIZE
        epoch_loss += predicted_loss.item()

        if i_count % PLT_INTERVAL == 0 and i_count != 0:
            with torch.no_grad():
                idx = np.random.randint(test_data_start, test_data_end-1)
                x_sample = original_data[idx].unsqueeze(0).to(DEVICE)
                x_noisy = noisy_data[idx].unsqueeze(0).to(DEVICE)
                x_pred = model(x_noisy)

                err = torch.nn.functional.mse_loss(x_pred, x_sample)
                vol = x_sample.shape[0]*x_sample.shape[1]*x_sample.shape[2]*x_sample.shape[3]
                accuracy = 1 - err.item()/vol
                plot_loss_accuracy(epoch, episode, epoch_loss/e_count, accuracy, prefix="noise")
                plot_noisy_reconstructions(epoch, episode, x_sample.squeeze(0).cpu().numpy(), x_pred.squeeze(0).cpu().numpy(), x_noisy.squeeze(0).cpu().numpy(), 10, prefix="noise")

        print(f"EPISODE {episode+1} LOSS: {predicted_loss.item()/BATCH_SIZE} SAMPLE_ACCURACY: {accuracy}")

    if epoch % SAVE_INTERVAL == 0:
        save_model(epoch, episode, optim, model, prefix="noise")
        save_epoch_data(epoch, episode, epoch_loss/i_count, accuracy, prefix="noise")
