from PIL import Image
import cv2 as cv
import os

##
## SCRIPT not yet functional
##


def upscale_pix_values(target_filename):
    #pre targets
    if 'pre' in target_filename:
        target_path_complete = 'train/' + 'targets/' + target_filename
        img = cv.imread(target_path_complete)
        th, threshimg = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
        cv.imwrite('train/' + 'new_targets/' + target_filename, threshimg)
    #post targets
    else:
        target_path_complete = 'train/' + 'targets/' + target_filename
        img = Image.open(target_path_complete)
        imgage = img.load()
        for x in range(1024):
            for y in range(1024):
                newvalue = imgage[x,y] * 100
                Image.putpixel((x, y), newvalue)
        Image.save('train/' + 'new_targets/' + target_filename)


if __name__ == '__main__':
    target_filenames = os.listdir('train/targets')
    for target_filename in target_filenames:
        upscale_pix_values(target_filename)
