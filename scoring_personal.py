'''
This is my personal scoring system
If counts the amount of pixel which have been predicted right
It is different to the xview2 scoring system because it does not account for
true positive predictions etc. This is beneficial since early stages 
of the trianed net might not hit true positives but do predict patches
of pixels right. Also it skips the damage predictions in the scoring folder
'''

import os
import numpy as np
from PIL import Image
import cv2 as cv

class scorer:
    def __init__(self, predictionPath, gtTargetPath):
        self.predictionPath = predictionPath
        self.gtTargetPath = gtTargetPath
        
        self.pred_img_clean = []
        self.gt_img_clean = []

        pred_not_cleaned = sorted(os.listdir(self.predictionPath))
        for x in range(len(pred_not_cleaned)):
            if 'localization' in pred_not_cleaned[x]:
                self.pred_img_clean.append(pred_not_cleaned[x])
        gt_not_cleaned = sorted(os.listdir(self.gtTargetPath))
        for x in range(len(gt_not_cleaned)):
            if 'localization' in gt_not_cleaned[x]:
                self.gt_img_clean.append(gt_not_cleaned[x])

        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.true_neg = 0

        #check if prediction and targets have same amount of items
        #assert len(predictionPath) == len(gtTargetPath), 'predictions and Targets do not have the same amount of items'

    def __getitem__(self, index):
        prediction = self.pred_img_clean[index]
        gtTarget = self.gt_img_clean[index]
        return prediction, gtTarget

    def score(self):
        counter = 0
        for index in range(len(self.pred_img_clean)):
            if counter > 100:
                continue
            counter += 1
            prediction, gtTarget = self.__getitem__(index)
            print(index, prediction, gtTarget)
            #read images as greyscale to make comparison easier
            pred_img = cv.imread(self.predictionPath+ '/' + prediction, 0)
            gt_img = cv.imread(self.gtTargetPath+ '/' + gtTarget, 0)

            #check if prediction matches ground truth
            for x in range(1024):
                for y in range(1024):
                    pred_pix_value = pred_img[x,y] 
                    gt_pix_value = gt_img[x,y]
                    #if gt_pix_value == 1:
                    if pred_pix_value == 1 and gt_pix_value == 1:
                        self.true_pos += 1
                    elif pred_pix_value == 1 and gt_pix_value == 0:
                        self.false_pos += 1
                    elif pred_pix_value == 0 and gt_pix_value == 0:
                        self.true_neg += 1
                    elif pred_pix_value == 0 and gt_pix_value == 1:
                        self.false_neg += 1
                    else:
                        print('pixelvalues did not match any of the cases(true_pos, true_neg ...) ')
        total_pixels = self.true_pos + self.true_neg + self.false_pos + self.false_neg
        return(total_pixels, self.true_pos, self.true_neg, self.false_pos, self.false_neg)

if __name__ == '__main__':
    Scorer = scorer('predictionT31/predictions', 'predictionT31/gt-targets')
    totalpix, tp, tn, fp, fn = Scorer.score()
    print (totalpix, tp, tn, fp, fn)
    per = tp / (tp+fp)
    rec = tp / (tp + fn)
    f1 = 2 * ((per * rec)/(per + rec))
    print ('f1:', f1)
    one_percent_total = (totalpix / 100)
    percent_of_tp = tp / one_percent_total
    percent_of_tn = tn / one_percent_total
    percent_of_fp = fp / one_percent_total
    percent_of_fn = fn / one_percent_total
    print('TP :', tp, 'FN ;', fn, 'FP :', fp)
    print('percent tp', percent_of_tp, 'percent_of_tn', percent_of_tn, 'percent_of_fp', percent_of_fp, 'percent_of_fn', percent_of_fn)