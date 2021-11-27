#%%
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv


#%%
def magic_wand(img_hsv,color,tolerance):
    mask = np.zeros((len(img_hsv),len(img_hsv[0])))
    for i in range(len(img_hsv)):
        for j in range(len(img_hsv[0])):
            keep = True
            for k in range(len(img_hsv[0][0])):
                if abs(img_hsv[i][j][k]-color[k]) > tolerance:
                    keep = False
            if keep:
                mask[i][j] = 1

    return mask



def hsv_degre_and_percent_to_255(hsv_degre_and_percent):
    hsv_255 = [0,0,0]
    hsv_255[0] = hsv_degre_and_percent[0]*255/360
    hsv_255[1] = hsv_degre_and_percent[1]*255/100
    hsv_255[2] = hsv_degre_and_percent[2]*255/100
    return hsv_255


def hsv_255_to_degre_and_percent(hsv_255):
    hsv_degre_and_percent = [0,0,0]
    hsv_degre_and_percent[0] = hsv_255[0]*360/255
    hsv_degre_and_percent[1] = hsv_255[1]*100/255
    hsv_degre_and_percent[2] = hsv_255[2]*100/255
    return hsv_degre_and_percent




#%%
file_path = 'D:/Documents/MSC/machine vision/project/MinneApple/imgs/detection/train/images/20150919_174151_image1.png'
# Loads an image
src = cv.imread(file_path)
# Check if image is loaded fine
src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


blue_color_hsv_deg_and_percent = [211, 18, 93]
blue_color_hsv_255 = hsv_degre_and_percent_to_255(blue_color_hsv_deg_and_percent)

yellow_apple_color_hsv_deg_and_percent = [58, 75, 100]
yellow_apple_hsv_255 = hsv_degre_and_percent_to_255(yellow_apple_color_hsv_deg_and_percent)

brown_color_hsv_deg_and_percent = [36, 78, 37]
brown_color_hsv_255 = hsv_degre_and_percent_to_255(brown_color_hsv_deg_and_percent)

dark_green_color_hsv_deg_and_percent = [103, 71, 13]
dark_green_color_hsv_255 = hsv_degre_and_percent_to_255(dark_green_color_hsv_deg_and_percent)


mask_blue = magic_wand(src_hsv,blue_color_hsv_255,80)
mask_yellow_apple = magic_wand(src_hsv,yellow_apple_hsv_255,60)
mask_brown = magic_wand(src_hsv,brown_color_hsv_255,50)
mask_dark_green = magic_wand(src_hsv,dark_green_color_hsv_255,50)

#%%
plt.figure(figsize = (10,10))
plt.imshow(mask_blue,cmap= "gray")
plt.show()

plt.figure(figsize = (10,10))
plt.imshow(mask_brown,cmap= "gray")
plt.show()

plt.figure(figsize = (10,10))
plt.imshow(mask_dark_green,cmap= "gray")
plt.show()

plt.figure(figsize = (10,10))
plt.imshow(np.invert(np.logical_or(np.logical_or(mask_brown,mask_blue),mask_dark_green)),cmap= "gray")
plt.show()

plt.figure(figsize = (10,10))
plt.imshow(mask_yellow_apple,cmap= "gray")
plt.show()


#%%