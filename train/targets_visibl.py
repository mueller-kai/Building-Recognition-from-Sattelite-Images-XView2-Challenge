import image
from PIL import Image

im = Image.open('/Volumes/externe_ssd/train/targets/guatemala-volcano_00000019_post_disaster_target.png')
pix = im.load()

#multiply pixelvalues by 100
#it will show the three different pixelvalues used in the target.pngs
for x in range(1024):
    for y in range(1024):
        newvalue = pix[x,y] * 100
        im.putpixel((x, y), newvalue)

im.save("new_shaded.png")