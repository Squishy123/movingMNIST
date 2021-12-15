import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path((Path(__file__).parent / '../').resolve())
RESULTS_PATH = Path(str(ROOT) + "/results/")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

loss_fig, (loss_plt, accuracy_plt) = plt.subplots(1, 2)


def plot_loss_accuracy(epoch, episode, loss, accuracy):
    loss_plt.plt((epoch + 1) * (episode + 1), loss)
    accuracy_plt((epoch + 1) * (episode + 1), accuracy)
    loss_fig.savefig(str(RESULTS_PATH) + "/loss_accuracy.png")


# def plot_reconstructions()
