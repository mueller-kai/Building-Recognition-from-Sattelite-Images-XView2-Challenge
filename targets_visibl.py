from PIL import Image
import cv2 as cv
import os
import config


class targetHandler:
    """class that handles target transformations"""

    def __init__(self, target_filenames) -> None:
        self.target_filenames = target_filenames

    def upscale_pix_values(self):
        for target_filename in self.target_filenames:
            # pre targets
            if "pre" in target_filename:
                target_path_complete = "train/" + "targets/" + target_filename
                img = Image.open(target_path_complete)
                image = img.load()
                for x in range(1024):
                    for y in range(1024):
                        if image[x, y] == 1:
                            image[x, y] = 255
                img.save("train/" + "new_targets/" + target_filename)
            # post targets
            else:
                target_path_complete = "train/" + "targets/" + target_filename
                img = Image.open(target_path_complete)
                image = img.load()
                for x in range(1024):
                    for y in range(1024):
                        newvalue = image[x, y] * 100
                        image[x, y] = newvalue
                img.save("train/" + "new_targets/" + target_filename)

    def downscale_pix_values(self):
        for target_filename in self.target_filenames:
            target_path_complete = "scoring/" + "predictions/" + target_filename
            img = Image.open(target_path_complete)
            image = img.load()
            for x in range(1024):
                for y in range(1024):
                    pixelimgvalue = image[x, y]
                    # pixelvalue = (pixelimg[0] + pixelimg[1] + pixelimg[2] ) /3
                    if pixelimgvalue == 255:
                        image[x, y] = 1
            img.save("scoring/" + "predictions/" + target_filename)

    def count_amount_pix_houses(self):
        """function returns housepixel ratio in Dataset"""
        print(self.target_filenames)
        house_pixel = 0
        background_pixel = 0
        for target_filename in self.target_filenames:

            # only calculate pixels in pre images
            if "pre" in target_filename:
                target_path_complete = "train/" + "targets/" + target_filename
                img = Image.open(target_path_complete)
                image = img.load()
                print(target_filename)

                # itterate over pic
                for x in range(config.INPUT_IMAGE_HEIGHT):
                    for y in range(config.INPUT_IMAGE_WIDTH):
                        pixelimg = image[x, y]
                        if pixelimg > 0:
                            house_pixel += 1
                        else:
                            background_pixel += 1

        house_pixel_ratio = (house_pixel + background_pixel) / house_pixel
        print(f"house_pixel_ratio: {house_pixel_ratio}, house_pixel {house_pixel}, background_pixel {background_pixel}")
        return house_pixel_ratio


if __name__ == "__main__":
    '''
    Use this file to perform target trainsformations
    Make sure to adopt the path variables in the file for your needs
    '''
    
    target_filenames = os.listdir("train/targets")
    handler = targetHandler(target_filenames)
    handler.count_amount_pix_houses()
