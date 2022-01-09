import torch
import os

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 4
BATCH_SIZE = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_IMAGE_HEIGHT = 1024
INPUT_IMAGE_WIDTH = 1024

BASE_OUTPUT = "output"
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_disaster_vision.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

TEST_PATHS = os.path.sep.join(["test", "images"])
TEST_PATHS = os.listdir(TEST_PATHS)
for idx, image in enumerate(TEST_PATHS):
    TEST_PATHS[idx] = 'test/'+ 'images/'+ image

TEST_PATHS = [x for x in TEST_PATHS if "pre" in x]

GROUNDTRUTH_MASKS = os.path.sep.join(["test", "new_targets"])

# define threshold to filter weak predictions
THRESHOLD = 0.5

NUMBER_PICS_IN_DATASET = 20