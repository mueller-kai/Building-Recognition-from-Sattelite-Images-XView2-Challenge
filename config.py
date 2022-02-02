import torch
import os

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 200
BATCH_SIZE = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

INPUT_IMAGE_HEIGHT = 1024
INPUT_IMAGE_WIDTH = 1024

BASE_OUTPUT = "output"
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_disaster_vision_BCEWLL_200_newT.pthe210_.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

TRAIN_PATHS = os.path.sep.join(["train", "images"])
TRAIN_PATHS = os.listdir(TRAIN_PATHS)
for idx, image in enumerate(TRAIN_PATHS):
    TRAIN_PATHS[idx] = 'train/'+ 'images/'+ image
#
TRAIN_PATHS = [x for x in TRAIN_PATHS if "pre" in x]

VALIDATION_PATHS = os.path.sep.join(["validation", "images"])
VALIDATION_PATHS = os.listdir(VALIDATION_PATHS)
for idx, image in enumerate(VALIDATION_PATHS):
    VALIDATION_PATHS[idx] = 'validation/'+ 'images/'+ image
VALIDATION_PATHS = [x for x in VALIDATION_PATHS if "pre" in x]

TEST_PATHS = os.path.sep.join(["test", "images"])
TEST_PATHS = os.listdir(TEST_PATHS)
for idx, image in enumerate(TEST_PATHS):
    TEST_PATHS[idx] = 'test/'+ 'images/'+ image
TEST_PATHS = [x for x in TEST_PATHS if "pre" in x]

GROUNDTRUTH_MASKS = os.path.sep.join(["test", "new_targets"])

# define threshold to filter weak predictions
THRESHOLD = 0.31

NUMBER_PICS_IN_DATASET = 800