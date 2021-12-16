from moving_mnist_dataset import MovingMNISTDataset
from context_encoder import ContextAutoencoder
from callbacks import plot_loss_accuracy, plot_reconstructions, save_epoch_data, save_model

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch
import numpy as np

NUM_FRAMES = 1
BATCH_SIZE = 1000
TOTAL_EPOCHS = 500
SAVE_INTERVAL = 10000
PLT_INTERVAL = 40000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

original_data = MovingMNISTDataset(num_frames=NUM_FRAMES)
#print(original_data[0].shape)
'''
print(original_data[0:100].shape)
fig, (a, b, c) = plt.subplots(1, 3)
a.imshow(original_data[0][1])
b.imshow(original_data[0][2])
c.imshow(original_data[0][3])
plt.show()
exit()
'''
# print(len(original_data))
#rint(original_data[60000].shape)
# exit()

#training_data = original_data[:int(len(original_data)*0.9)]
#test_data = original_data[int(len(original_data)*0.9):]
training_data_start = 0
training_data_end = int(len(original_data)*0.9)
test_data_start = int(len(original_data)*0.9)
test_data_end = len(original_data)

model = ContextAutoencoder(channels=10-NUM_FRAMES+1).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
print("BEGINNING TRAINING")

for epoch in range(TOTAL_EPOCHS):
    print(f"STARTING EPOCH: {epoch+1}")

    epoch_loss = 0
    accuracy = 0
    i_count = 0
    for episode in range((training_data_end-training_data_start)//BATCH_SIZE):
        optim.zero_grad()

        current_state = original_data[episode*BATCH_SIZE:(episode+1)*BATCH_SIZE].to(DEVICE)
        computed_state = model(current_state)

        predicted_loss = torch.nn.functional.mse_loss(computed_state, current_state)

        predicted_loss.backward()

        for param in model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()

        i_count += BATCH_SIZE
        epoch_loss += predicted_loss.item()

        print(f"EPISODE {episode+1} LOSS: {predicted_loss.item()/BATCH_SIZE}")

        if i_count % PLT_INTERVAL == 0 and i_count != 0:
            with torch.no_grad():
                x_sample = original_data[np.random.randint(test_data_start, test_data_end-1)].unsqueeze(0).to(DEVICE)
                x_pred = model(x_sample)

                err = torch.nn.functional.mse_loss(x_pred, x_sample)
                accuracy = -err.item()
                plot_loss_accuracy(epoch, episode, epoch_loss/i_count, accuracy)
                plot_reconstructions(epoch, episode, x_sample.squeeze(0).cpu().numpy(), x_pred.squeeze(0).cpu().numpy(), 10-NUM_FRAMES+1)

    #if i_count % SAVE_INTERVAL == 0:
    # SAVE EVERY EPOCH
    save_model(epoch, episode, optim, model)
    save_epoch_data(epoch, episode, epoch_loss/i_count, accuracy)

   
    
