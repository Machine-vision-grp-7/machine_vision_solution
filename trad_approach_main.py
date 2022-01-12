#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
from os import listdir
from os.path import isfile, join
import statistics
from numpy.core.numeric import outer
import sys

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

import sys

PATH_TO_PROJECT_FOLDER = 'D:/Documents/MSC/machine vision/project/' 
sys.path.insert(0, 'D:/Documents/MSC/machine vision/project/')
from trad_approach_def import *

# %%    do only one image
path_imgs = "MinneApple/imgs/detection/train/images/"
path_masks = "MinneApple/imgs/detection/train/masks/"

parameters = [[57, 69, 69], 1, 3]

red_pic = "20150921_131346_image211.png"
weird_green_pic = "20150919_174730_image1331.png"
yellow_green_pic = "20150919_174151_image1.png"
green_pic = "20150919_174151_image231.png"

#choose here which picture to try
img_path = path_imgs + green_pic
mask_path = path_masks + green_pic

#show the original picture and mask
show_img(img_path,title = "The original picture")
show_img(mask_path,title = "The original mask")

#check the original mask and return "truth", a list holding
#informations about the apples, and "true_mask", an array being
#the mask from the database
truth,true_mask = ground_truth(mask_path,show = False)

#run the algorithm to return "found", a list holding
#informations about the apples, and "found_mask", an array being
#the generated mask of the algorithm
found,found_mask = count_apples(img_path,parameters,show = True,tech = "sum",opening_yellow=3)

#show the generated array, and circle the estimated position
#of the apples
show_img(found_mask,title = "The combination of the red and yellow mask")
show_img(generate_cool_mask(found_mask),title = "Our mask with circled apples")
show_img(draw_on_original(found_mask,img_path),title = "Original picture with circled apples")
print("they are", truth[0], "appless but we found",found[0])



#%% test variying parameters

L_results = []
i = 0


for tolerance in [45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]:
    for SV_tolerance in [65,66,67,68,69,70]:
        for opening_base in [1]:
            for opening_yellow in [3]:
                for s_kernel in [3]:
                    parameters = [[tolerance,SV_tolerance,SV_tolerance],opening_base,s_kernel]
                    print(parameters,opening_yellow)
                    statistics_of_batch = do_n_img(100,parameters,opening_yellow,show=False)
                    L_results.append((parameters,opening_yellow,statistics_of_batch))

print(L_results)

#%% use this to run the algorithm on the dataset:
L_truth,L_found,coucou,S_yi_minus_ychapi_squared,dif,MAE,RSME,R2 = do_n_img(570,[[57, 69, 69], 1, 3], 3)

# %%
#use this to get info on the database
L_true_masks, L_truth, L_names,L_etiquette,L_height,L_all_truth \
    = ground_truth_for_all_data()
