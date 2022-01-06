#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
from os import listdir
from os.path import isfile, join
import statistics
from numpy.core.numeric import outer
import sys


import sys
sys.path.insert(0, 'D:/Documents/MSC/machine vision/project/')
from trad_approach_def import *

# %%    do only one image
path_imgs = "MinneApple/imgs/detection/train/images/"
path_masks = "MinneApple/imgs/detection/train/masks/"

parameters = [[25,80,80],1,3]

red_pic = "20150921_131346_image211.png"
weird_green_pic = "20150919_174730_image1331.png"
yellow_green_pic = "20150919_174151_image1.png"
green_pic = "20150919_174151_image231.png"

#choose here which picture to try
img_path = path_imgs + red_pic
mask_path = path_masks + red_pic

#show the original picture and mask
show_img(img_path)
show_img(mask_path)

#check the original mask and return "truth", a list holding
#informations about the apples, and "true_mask", an array being
#the mask from the database
truth,true_mask = ground_truth(mask_path,show = False)

#run the algorithm to return "found", a list holding
#informations about the apples, and "found_mask", an array being
#the generated mask of the algorithm
found,found_mask = count_apples(img_path,parameters,show = True,tech = "sum",opening_yellow=3)
print(truth[0],found[0])

#show the generated array, and circle the estimated position
#of the apples
show_img(found_mask)
show_img(generate_cool_mask(found_mask))
show_img(draw_on_original(found_mask,img_path))


#%% use this to run the algorithm on the dataset:

Results = do_n_img(670,[[30, 80, 80], 2, 3], 3)
#Results =  (total nb of apple,error in apples,MAE,RSME,R2)
#%% test variying parameters

L_results = []
i = 0

for opening_i in [4]: #4 ! 
    for tolerance in [30]:
        for SV_tolerance in [80]:
            for opening_base in [0,1,2,3]:
                for opening_yellow in [0,1,2,3]:
                    for s_kernel in [1,2,3,4]:
                        parameters = [[tolerance,SV_tolerance,SV_tolerance],opening_base,s_kernel]
                        print(parameters,opening_yellow)
                        statistics_of_batch = do_n_img(100,parameters,opening_yellow)
                        L_results.append((parameters,opening_i,statistics_of_batch))
                
print(L_results)
#%% best ones

L_results_sorted = sorted(L_results, key = lambda x: x[2][3])
bests = L_results_sorted[-2:]
for i in  bests:
    print(i)
#best parameters for now :
#[[30, 80, 80], 2, 3], 3
#[[30, 80, 80], 2, 3], 4
#%%
do_n_img(670,[[30, 80, 80], 2, 3], 3)
# %%
#use this to get info on the database
L_true_masks, L_truth, L_names,L_etiquette,L_height \
    = ground_truth_for_all_data()
