import torch

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_IMAGE_HEIGHT = 1024
INPUT_IMAGE_WIDTH = 1024
