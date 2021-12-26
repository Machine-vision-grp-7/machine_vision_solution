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
default_file = 'D:/Documents/MSC/machine vision/project/MinneApple/imgs/detection/train/images/20150919_174151_image1.png'
# Loads an image
src = cv.imread(default_file)
# Check if image is loaded fine
src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


blue_color_to_keep_hsv_deg_and_percent = [211, 18, 93]
blue_color_to_keep_hsv_255 = hsv_degre_and_percent_to_255(blue_color_to_keep_hsv_deg_and_percent)

yellow_apple_color_to_keep_hsv_deg_and_percent = [58, 75, 100]
yellow_apple_to_keep_hsv_255 = hsv_degre_and_percent_to_255(yellow_apple_color_to_keep_hsv_deg_and_percent)

brown_color_to_keep_hsv_deg_and_percent = [36, 78, 37]
brown_color_to_keep_hsv_255 = hsv_degre_and_percent_to_255(brown_color_to_keep_hsv_deg_and_percent)

dark_green_color_to_keep_hsv_deg_and_percent = [103, 71, 13]
dark_green_color_to_keep_hsv_255 = hsv_degre_and_percent_to_255(dark_green_color_to_keep_hsv_deg_and_percent)


mask_blue = magic_wand(src_hsv,blue_color_to_keep_hsv_255,80)
mask_yellow_apple = magic_wand(src_hsv,yellow_apple_to_keep_hsv_255,60)
mask_brown = magic_wand(src_hsv,brown_color_to_keep_hsv_255,50)
mask_dark_green = magic_wand(src_hsv,dark_green_color_to_keep_hsv_255,50)

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
hres_img = cv.adaptiveThreshold(src_hsv,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,301,3)



plt.figure(figsize = (10,10))
plt.imshow(hres_img,cmap= "gray")
plt.show()


rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                            param1=100, param2=30,
                            minRadius=10, maxRadius=35)


if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(src, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(src, center, radius, (255, 0, 255), 3)

src = src[:,:,::-1] #RGB -> BGR
# plt.figure(figsize = (20,20))
# plt.imshow(src)
# plt.show()

# %%

# %%


kernel = np.ones((3,3),np.uint8)
# closing
closing = cv.morphologyEx(thres_img, cv.MORPH_CLOSE, kernel, iterations=2)

# erosion
kernel = np.ones((3,3),np.uint8)
erosion = cv.erode(thres_img,kernel,iterations = 2)

# closing
kernel = np.ones((3,3),np.uint8)
closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel, iterations=2)

# erosion
kernel = np.ones((3,3),np.uint8)
erosion2 = cv.erode(closing,kernel,iterations = 2)

# remove boarder pixels
erosion2[:50, :] = 0
erosion2[:, :50] = 0
erosion2[-50:, :] = 0
erosion2[:, -50:] = 0

kernel = np.ones((1,1),np.uint8)
opening = cv.morphologyEx(erosion2, cv.MORPH_OPEN, kernel,iterations = 1)

plt.figure(figsize = (15,20))
plt.subplot(131)
plt.imshow(src)
plt.subplot(132)
plt.imshow(thres_img,cmap="gray")
plt.subplot(133)
plt.imshow(opening,cmap="gray")
plt.show()
# %%
nb_components, output, stats, centroids =\
     cv.connectedComponentsWithStats(opening, connectivity=8)
print(nb_components, output, stats, centroids)
# %%
#%%
src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

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
def average_color(img_path,mask_path):
    image = cv.imread(img_path)
    mask = cv.imread(mask_path)
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    new_image = np.array(image.shape)
    show_img(img_path)
    show_img(mask)
    avg = [0,0,0]
    sum = 0
    for i in range(len(image_hsv)):
        for j in range(len(image_hsv[0])):
            if list(mask[i][j]) != [0, 0, 0]:
                for k in range(len(image_hsv[0][0])):
                    avg[k] += image_hsv[i][j][k]
                sum+=1
                new_image[i][j] = image[i][j]
    
    avg[0] = avg[0]/sum
    avg[1] = avg[1]/sum
    avg[2] = avg[2]/sum
    print(sum)
    plt.figure(figsize = (10,10))
    plt.title("Image")
    plt.imshow(new_image)
    print(new_image)
    plt.show()
    return hsv_255_to_degre_and_percent(avg)

print(average_color(img_path,mask_path))