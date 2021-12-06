from PIL import Image

im_pre = Image.open('train/images/socal-fire_00000631_pre_disaster.png')
im_post = Image.open('train/images/socal-fire_00000631_post_disaster.png')
target = Image.open('train/targets/socal-fire_00000631_post_disaster_target.png')



#calculate new pixel values if target indicates a building otherwise pix = black
def calculate_diff_targetarea (image_pre, image_post, target):
    
    #load images
    imgpre = image_pre.load()
    imgpost = image_post.load()
    tar = target.load()
    width, height = image_post.size

    #create new image for difference values in greyscale
    difference_image = Image.new('L', (width, height))

    for x in range(height):
        for y in range(width):
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

if __name__ == "__main__":
    calculate_diff_targetarea(im_pre, im_post, target)