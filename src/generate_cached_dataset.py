from pathlib import Path
import torch
import numpy as np

from moving_mnist_dataset import MovingMNISTDataset

import sys
import getopt


def main(argv):
    BATCH_SIZE = 1000
    TOTAL_EPOCHS = 500
    PLT_INTERVAL = 50000
    SAVE_INTERVAL = 100000
    NUM_FRAMES = 5

    OUTPUT_FILE = "mnist_seq.npz"
    DEFAULT_DATASET_LOCATION = Path((Path(__file__).parent / '../datasets/movingMNIST/').resolve())
    DEFAULT_DATASET_LOCATION.mkdir(parents=True, exist_ok=True)

    try:
        opts, args = getopt.getopt(argv, "hf:o:", ["numframes=", "ofile="])
    except getopt.GetoptError:
        print('test.py -f <num_frames> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -f <numframes> -o <outputfile>')
            print("ERROR")
            sys.exit()
        elif opt in ("-f", "--numframes"):
            NUM_FRAMES = int(arg)
        elif opt in ("-o", "--ofile"):
            OUTPUT_FILE = "/" + str(arg)

    dataset = MovingMNISTDataset()

    preloaded_curr_state_ALL = None
    preloaded_next_state_ALL = None

    for i in range(len(dataset)//BATCH_SIZE):
        print(f"{i}/{len(dataset)//BATCH_SIZE}")

        data = dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        preloaded_curr_state = data[:, :]
        print(preloaded_curr_state.shape)
        preloaded_curr_state = preloaded_curr_state.unfold(1, NUM_FRAMES, 1)
        print(preloaded_curr_state.shape)
        exit()
        #preloaded_curr_state = torch.reshape(preloaded_curr_state, (preloaded_curr_state.shape[0], preloaded_curr_state.shape[1], NUM_FRAMES*64, 10, 10))
        # print(preloaded_curr_state.shape)
        '''
        preloaded_next_state = data[:, 1:]
        preloaded_next_state = preloaded_next_state.unfold(1, NUM_FRAMES, 1)
        preloaded_next_state = torch.reshape(preloaded_next_state, (preloaded_next_state.shape[0], preloaded_next_state.shape[1], NUM_FRAMES*64, 10, 10))
        '''
        # SAVING PRELOADED DATA
        np.savez(str(DEFAULT_DATASET_LOCATION) + "/" + f"CACHE_{i}_" + OUTPUT_FILE, curr_state=preloaded_curr_state)  # next_state=preloaded_next_state)


if __name__ == "__main__":
    main(sys.argv[1:])
