import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/Volumes/externe_ssd/train/targets/guatemala-volcano_00000019_post_disaster_target.png')

#all pixels with a values above 0 will be set to 255
th, threshimg = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
cv.imwrite('new_binary.png', threshimg)