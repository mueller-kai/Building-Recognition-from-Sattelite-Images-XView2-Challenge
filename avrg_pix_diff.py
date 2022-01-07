from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

#calculate new pixel values if target indicates a building otherwise pix = black
def generate_diff_targetarea (image_pre, image_post, target, bool):
    #load images
    imgpre = image_pre.load()
    imgpost = image_post.load()
    tar = target.load()
    width, height = image_post.size

    #create new image for difference values in greyscale
    difference_image = Image.new('L', (width, height))

    #itterate over pixels of new difference image
    for x in range(height):
        for y in range(width):
            #only calculate new pixel values if the target is indicating a house
            if tar[x,y] != 0:
                #read values
                r1, g1, b1 = imgpre[x,y]
                r2, g2, b2 = imgpost[x,y]
                
                #calculate diff
                diff_red = abs (r1 - r2)
                diff_blue = abs (b1 - b2)
                diff_green = abs (g1 - g2)

                #average difference
                avg_diff = (diff_blue + diff_green + diff_red) / 3

                difference_image.putpixel((x,y), int(avg_diff))
            else:
                difference_image.putpixel((x,y), 0)
    
    difference_image.save("pixel_diffs.png")
    return

def calculate_diff_target_area_for_image(image_pre, image_post, target) -> dict:
    # This function calculated the differences in pixel values for the area of a house.
    # It outputs a dict that contains the calculated avrg. pixeldiff.
    # For the area of a house sorted by destruction level

    image_pre = Image.open(f'{image_pre}')
    image_post = Image.open(f'{image_post}')
    target = Image.open(f'{target}')
    
    # load images
    imgpre = image_pre.load()
    imgpost = image_post.load()
    tar = target.load()
    width, height = image_post.size
    # this will contain the average pixel differences between pre and post image on a given position (x,y)
    # sorted by the level of distruction of the house
    out = {
        1: [],
        2: [],
        3: [],
        4: [],
    }

    # itterate over pixels of new difference image
    for x in range(height):
        for y in range(width):
            distroyment_value = tar[x,y]
            # only calculate new pixel values if the target is indicating a house
            if distroyment_value != 0:
                # read values
                r1, g1, b1 = imgpre[x,y]
                r2, g2, b2 = imgpost[x,y]
                
                # calculate diff
                diff_red = abs (r1 - r2)
                diff_blue = abs (b1 - b2)
                diff_green = abs (g1 - g2)

                #average difference
                avg_diff = (diff_blue + diff_green + diff_red) / 3

                out[distroyment_value].append(avg_diff)
    
    return out

def get_sorted_items_from_dataset():
    # A class of which one dataset is one object
    # Each dataset object has a function to get one item of this dataset
    
    image_folder = 'train/images'
    target_folder = 'train/targets'
    labels_folder = 'train/labels'


    images_paths_pre = []
    images_paths_post = []

    target_paths_pre = []
    target_paths_post = []

    labels_paths_pre = []
    labels_paths_post = []

    images_filenames = os.listdir(image_folder)

    for img_fn in images_filenames:
        # ignore mac files
        if img_fn.startswith('.'):
            continue

        # remove post desaster images because everthing is done relative to the pre disaster images (filenames are beeing altered to post thats it)
        if 'post_disaster' in img_fn:
            continue

        # add pre desaster paths for target, image and label
        images_paths_pre.append(os.path.join(image_folder, img_fn))

        pre_target_fn = img_fn.replace('.png', '_target.png')
        target_paths_pre.append(os.path.join(target_folder, pre_target_fn))

        label_fn = img_fn.replace('.png', '.json')
        labels_paths_pre.append(os.path.join(labels_folder, label_fn))

        # add post desaster paths for target image and label (replace pre with post)
        post_img_fn = img_fn.replace('pre_disaster', 'post_disaster')
        images_paths_post.append(os.path.join(image_folder, post_img_fn))

        post_target_fn = post_img_fn.replace('.png', '_target.png')
        target_paths_post.append(os.path.join(target_folder, post_target_fn))

        label_fn = post_img_fn.replace('.png', '.json')
        labels_paths_post.append(os.path.join(labels_folder, label_fn))
    
    return images_paths_pre, images_paths_post, target_paths_pre, target_paths_post, labels_paths_pre, labels_paths_post

def get_one_item(index):
    images_paths_pre, images_paths_post, target_paths_pre, target_paths_post, labels_paths_pre, labels_paths_post = get_sorted_items_from_dataset()

    image_fn_pre = images_paths_pre[index]
    image_fn_post = images_paths_post[index]

    target_fn_pre = target_paths_pre[index]
    target_fn_post = target_paths_post[index]

    #label_fn = labels_paths[index]

    return image_fn_pre, image_fn_post, target_fn_pre, target_fn_post
def calculate_diff_target_area_for_image(image_pre, image_post, target):

    image_pre = Image.open(f'{image_pre}')
    image_post = Image.open(f'{image_post}')
    target = Image.open(f'{target}')

    #load images
    imgpre = image_pre.load()
    imgpost = image_post.load()
    tar = target.load()
    width, height = image_post.size

    out = {
        1: [],
        2: [],
        3: [],
        4: [],
    }

    #itterate over pixels of new difference image
    for x in range(height):
        for y in range(width):
            distroyment_value = tar[x,y]
            #only calculate new pixel values if the target is indicating a house
            if distroyment_value != 0:
                #read values
                r1, g1, b1 = imgpre[x,y]
                r2, g2, b2 = imgpost[x,y]

                #calculate diff
                diff_red = abs (r1 - r2)
                diff_blue = abs (b1 - b2)
                diff_green = abs (g1 - g2)

                #average difference
                avg_diff = (diff_blue + diff_green + diff_red) / 3

                out[distroyment_value].append(avg_diff)

    return out

class DestasterVisionDataset(Dataset):
    def __init__(
        self,
        image_folder='train/images',
        target_folder='',
        labels_folder='',
        transforms = transforms
    ):

        self.images_paths_pre = []
        #self.images_paths_post = []

        self.target_paths_pre = []
        #self.target_paths_post = []

        #self.labels_paths_pre = []
        #self.labels_paths_post = []

        images_filenames = os.listdir(image_folder)

        for img_fn in images_filenames:
            # ignore mac files
            if img_fn.startswith('.'):
                continue

            # remove post desaster images
            if 'post_disaster' in img_fn:
                continue

            # add pre desaster paths
            self.images_paths_pre.append(os.path.join(image_folder, img_fn))
            pre_target_fn = img_fn.replace('.png', '_target.png')
            self.target_paths_pre.append(os.path.join(target_folder, pre_target_fn))
            #label_fn = img_fn.replace('.png', '.json')
            #self.labels_paths_pre.append(os.path.join(labels_folder, label_fn))

            # add post desaster paths
            #post_img_fn = img_fn.replace('pre_disaster', 'post_disaster')
            #self.images_paths_post.append(os.path.join(image_folder, post_img_fn))
            #post_target_fn = post_img_fn.replace('.png', '_target.png')
            #self.target_paths_post.append(os.path.join(target_folder, post_target_fn))
            #label_fn = post_img_fn.replace('.png', '.json')
            #self.labels_paths_post.append(os.path.join(labels_folder, label_fn))


    def __getitem__(self, index):
        image_fn_pre = self.images_paths_pre[index]
        image_fn_post = self.images_paths_post[index]

        target_fn_pre = self.target_paths_pre[index]
        target_fn_post = self.target_paths_post[index]

        #label_fn = self.labels_paths[index]

        return image_fn_pre, image_fn_post, target_fn_pre, target_fn_post

    def __len__(self):
        return len(self.images_paths)


def calculate_diff_target_area_for_dataset():

    destaster_vision_dataset = DestasterVision(
        image_folder='train/images',
        target_folder='train/targets',
        labels_folder='train/labels'
    )

    diff_all_images = {
        1: [],
        2: [],
        3: [],
        4: [],
    }

    num_items_in_desaster_dataset = len(destaster_vision_dataset.images_paths_pre)
    for index in range(100): #range(num_items_in_desaster_dataset):
        image_fn_pre, image_fn_post, target_fn_pre, target_fn_post = destaster_vision_dataset.get_item(index)
        diffs_single_image = calculate_diff_target_area_for_image(image_fn_pre, image_fn_post, target_fn_post)
        for key, value in diffs_single_image.items():
            diff_all_images[key] += value

    return diff_all_images

def plot_bar_chart(diff_all_images):
    # bar chart mit destroyment_value auf x-achse und pixel differenz auf y achse
    import matplotlib.pyplot as plt 

    for i in range(1,5):
        values = np.array(diff_all_images[i])
        # scatter values
        #num_diffs = len(diff_all_images[i])
        #plt.scatter([i] * num_diffs, diff_all_images[i])
        # plot median
        plt.scatter([i], np.median(np.array(diff_all_images[i])))
        print(f'Disaster score: {i}, median: {np.median(values):.2f}, mean: {np.mean(values):.2f}, standard deviation: {np.std(values):.2f}')

    plt.show()


if __name__ == "__main__":

    # to understand data, test function plotting images
    #im_pre = Image.open('train/images/socal-fire_00000631_pre_disaster.png')
    #im_post = Image.open('train/images/socal-fire_00000631_post_disaster.png')
    #target = Image.open('train/targets/socal-fire_00000631_post_disaster_target.png')
    #generate_diff_targetarea(im_pre, im_post, target)

    # compute difference for all images and create bar chart
    diff_all_images = calculate_diff_target_area_for_dataset()
    plot_bar_chart(diff_all_images)