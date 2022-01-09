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
#%% deal with results
L_work = L_results.copy()
for i in range(len(L_work)):
    L_work[i] = [L_work[i][0],L_work[i][1],round(L_work[i][2][3],3)]
    print(L_work[i])
#%% best ones
L_results_sorted = sorted(L_results, key = lambda x: x[2][3])
bests = L_results_sorted[:3]
for i in  bests:
    print(i)

#%% use this to run the algorithm on the dataset:

Results = do_n_img(20,[[30, 80, 80], 2, 3], 3)
#Results =  (total nb of apple,error in apples,MAE,RSME,R2)
#%%
do_n_img(670,[[57, 69, 69], 1, 3], 3)
#best parameters for now :
#[[30, 80, 80], 2, 3], 3
#[[30, 80, 80], 2, 3], 4

#[30, 80, 80], 2, 3], 4
#[30, 80, 80], 3, 3], 4
#[30, 80, 80], 3, 3], 4

#([[20, 50, 50], 4, 3], -1, (5867, -375, 9.370000000000001, 12.756566936288149, 0.8699305755546055))
#([[20, 50, 50], 3, 3], 0, (5867, -378, 9.38, 12.767928571228772, 0.8696987797389036))
#([[20, 50, 50], 2, 3], 1, (5867, -379, 9.39, 12.774584141959377, 0.8695628994331474))


# %%
#use this to get info on the database
L_true_masks, L_truth, L_names,L_etiquette,L_height \
    = ground_truth_for_all_data()

# %%
