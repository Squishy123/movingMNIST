import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path((Path(__file__).parent / '../').resolve())
RESULTS_PATH = Path(str(ROOT) + "/results/")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

loss_fig, (loss_plt, accuracy_plt) = plt.subplots(1, 2)


def plot_loss_accuracy(epoch, episode, loss, accuracy):
    loss_plt.scatter((epoch + 1) * (episode + 1), loss, color="red")
    accuracy_plt.scatter((epoch + 1) * (episode + 1), accuracy, color="blue")
    loss_fig.savefig(str(RESULTS_PATH) + "/loss_accuracy.png")


def plot_reconstructions(epoch, episode, actual, predicted, NUM_FRAMES=10):
    recon_fig, ax= plt.subplots(2, NUM_FRAMES, sharex='col', sharey='row')

    ax[0][NUM_FRAMES//2].set_title(f"Sample Reconstruction at Epoch {epoch+1}, Episode {episode+1}")

    for i in range(NUM_FRAMES):
        ax[0][i].imshow(actual[i])
        ax[1][i].imshow(predicted[i])

    recon_fig.savefig(str(RESULTS_PATH) + f"/reconstruction_{epoch+1}_{episode+1}.png")