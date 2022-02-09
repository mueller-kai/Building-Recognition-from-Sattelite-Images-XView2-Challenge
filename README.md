Disaster-Vision

The automated recognition of buildings is the first step towards an automated assessment of building
damage on the basis of satellite images in the event of a disaster, which in turn offers a time advantage
for the emergency forces on site. For the creation of such a system, a corresponding dataset was provided by the xView2 Challenge.

This Repo provides a training pipeline for a U-Net for building footprint recognition on sattelite imagery.
2 U-Netversions with similar F1 scores are provided within this repo. The U-Nets generalise on different
regions/ building types in the world. Testing scripts to evaluate F1 scores are also provided.

How to use:
 - To Use the U-Nets for classification you can load them from the .pth files in the U-Net folder like so
'''
torch.load("U-Net/U-Net2.0/unet2.0_disaster_vision_BCEWLL_.pth")
'''
 - Make sure to give 1024 x 1024 as input images
 - specifiy the folder of the input images in config.py under TEST_PATHS
 - For U-Net2.1 adjust the thershhold in config.py to 0.85
   For U-Net1.1 adjust the thershhold in config.py to 0.31
 - create a folder called f"predictionT{threshhold * 100}" with the subfolders
     - predictions
     - gt-targets
     - visible-predictions
     
 # Note that there will be damage predictions saved in the folder, you can ignore them. They contain the same pictures
 # as the localization predictions and are only needed to satisfy the scoring_xView2.py script

Score: 
U-Net2.1 and U-Net 1.1 with their specified threshholds reach an F1 score of 0.62
