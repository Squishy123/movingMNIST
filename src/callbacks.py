import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np

ROOT = Path((Path(__file__).parent / '../').resolve())
RESULTS_PATH = Path(str(ROOT) + "/results/")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = Path(str(ROOT) + "/weights/")
WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)

loss_fig, (loss_plt, accuracy_plt) = plt.subplots(1, 2)
loss_plt.set_title("Model Loss")
accuracy_plt.set_title("Model Accuracy")

epoch_loss = []
epoch_accuracy = []


def save_epoch_data(epoch, episode, loss, accuracy, prefix="default"):
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)

    np.savez(str(RESULTS_PATH) + "/" + prefix + "/epoch_data.npz", epoch_loss=epoch_loss, epoch_accuracy=epoch_accuracy)


def plot_loss_accuracy(epoch, episode, loss, accuracy, prefix="default"):
    loss_plt.scatter((epoch + 1) * (episode + 1), loss, color="red")
    accuracy_plt.scatter((epoch + 1) * (episode + 1), accuracy, color="blue")
    loss_fig.savefig(str(RESULTS_PATH) + "/" + prefix + "/loss_accuracy.png")


def plot_reconstructions(epoch, episode, actual, predicted, NUM_FRAMES=10, prefix="default"):
    recon_fig, ax = plt.subplots(2, NUM_FRAMES, sharex='col', sharey='row')

    ax[0][NUM_FRAMES//2].set_title(f"Sample Reconstruction at Epoch {epoch+1}, Episode {episode+1}")

    for i in range(NUM_FRAMES):
        ax[0][i].imshow(actual[i])
        ax[1][i].imshow(predicted[i])

    recon_fig.savefig(str(RESULTS_PATH) + "/" + prefix + f"/reconstruction_{epoch+1}_{episode+1}.png")
    plt.close(recon_fig)


def plot_noisy_reconstructions(epoch, episode, actual, predicted, noisy, NUM_FRAMES=10, prefix="default"):
    recon_fig, ax = plt.subplots(3, NUM_FRAMES, sharex='col', sharey='row')

    ax[0][NUM_FRAMES//2].set_title(f"Sample Noisy Reconstruction at Epoch {epoch+1}, Episode {episode+1}")

    for i in range(NUM_FRAMES):
        ax[0][i].imshow(actual[i])
        ax[1][i].imshow(noisy[i])
        ax[2][i].imshow(predicted[i])

    recon_fig.savefig(str(RESULTS_PATH) + "/" + prefix + f"/reconstruction_{epoch+1}_{episode+1}.png")
    plt.close(recon_fig)


def save_model(epoch, episode, optim, model, path=str(WEIGHTS_PATH), prefix="default"):
    torch.save({
        'epoch': epoch,
        'episode': episode,
        'optimizer_state_dict': optim.state_dict(),
        'model_state_dict': model.state_dict()
    }, path+"/" + prefix + f"/model_weight_{epoch+1}_{episode+1}.pth")
