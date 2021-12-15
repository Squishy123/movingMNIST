from moving_mnist_dataset import MovingMNISTDataset
from context_encoder import ContextAutoencoder
from callbacks import plot_loss_accuracy

import matplotlib.pyplot as plt
from torchvision import transforms, utils
import torch

NUM_FRAMES = 5
BATCH_SIZE = 100
TOTAL_EPOCHS = 100
SAVE_INTERVAL = 100
PLT_INTERVAL = 100

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

original_data = MovingMNISTDataset(num_frames=NUM_FRAMES)

'''
print(original_data[0:100].shape)
fig, (a, b, c) = plt.subplots(1, 3)
a.imshow(original_data[0][1])
b.imshow(original_data[0][2])
c.imshow(original_data[0][3])
plt.show()
exit()
'''

training_data = original_data[:int(len(original_data)*0.9)]
test_data = original_data[int(len(original_data)*0.9):]


model = ContextAutoencoder(channels=NUM_FRAMES).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
print("BEGINNING TRAINING")
for epoch in range(TOTAL_EPOCHS):
    print(f"STARTING EPOCH: {epoch+1}")

    i_count = 0
    for episode in range(len(training_data)//BATCH_SIZE):
        print("LOL")
        optim.zero_grad()

        current_state = training_data[episode*BATCH_SIZE:(episode+1)*BATCH_SIZE].to(DEVICE)
        computed_state = model(current_state)

        predicted_loss = torch.nn.functional.mse_loss(computed_state, current_state)

        predicted_loss.backward()

        for param in model.parameters():
            if param.grad != None:
                param.grad.data.clamp_(-1, 1)

        optim.step()

        print(f"EPISODE {episode+1} LOSS: {predicted_loss.item()/BATCH_SIZE}")

        # if i_count % PLT_INTERVAL == 0:
        #    plot_loss_accuracy(epoch, episode, predicted_loss.item()/BATCH_SIZE)

        i_count += BATCH_SIZE

    '''with torch.no_grad():
        x_sample = test_data[np.random.randint(0, len(test_data))]
        x_pred = model(x_sample)

        err = torch.nn.functional.MSELoss(x_pred, x_sample)
        plot_loss_accuracy(epoch, episode, epoch_loss/i_count, err.item())
    '''
