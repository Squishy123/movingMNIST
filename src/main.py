from moving_mnist_dataset import MovingMNISTDataset
from context_encoder import ContextAutoencoder

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch

NUM_FRAMES = 5
BATCH_SIZE = 1000
TOTAL_EPOCHS = 100
SAVE_INTERVAL = 100
PLT_INTERVAL = 100

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

original_data = MovingMNISTDataset(num_frames=NUM_FRAMES)

print(original_data[0][0].shape)
plt.imshow(original_data[0][0])
plt.show()
exit()

training_data = original_data[:int(len(original_data)*0.9)]
test_data = original_data[int(len(original_data)*0.9):]


model = ContextAutoencoder(channels=NUM_FRAMES).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
print("BEGINNING TRAINING")
for epoch in range(TOTAL_EPOCHS):
    print(f"STARTING EPOCH: {epoch}")
    for episode in range(len(training_data)//BATCH_SIZE):
        optim.zero_grad()

        current_state = training_data[episode*BATCH_SIZE:(episode+1)*BATCH_SIZE].to(DEVICE)
        computed_state = model(current_state)

        predicted_loss = torch.nn.functional.mse_loss(computed_state, current_state)

        predicted_loss.backward()

        for param in model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()

        print(f"EPISODE {episode} LOSS: {predicted_loss.item()/BATCH_SIZE}")
