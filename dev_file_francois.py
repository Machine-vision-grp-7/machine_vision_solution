#%%
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
from os import listdir
from os.path import isfile, join
import statistics

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

def ground_truth(file_path,show = True):
    true_mask_src = cv.imread(file_path)
    true_mask_gray = cv.cvtColor(true_mask_src, cv.COLOR_BGR2GRAY)
    true_mask_srcu8 = np.uint8(true_mask_gray)
    Components_Stats = cv.connectedComponentsWithStats(true_mask_srcu8)
    
    if show:
        plt.figure(figsize = (10,10))
        plt.title("Real mask")
        plt.imshow(true_mask_gray,cmap= "gray")
        plt.show()

    return Components_Stats,true_mask_gray

def count_apples(source_path,parameters=[60,1,5,1,3],show = True):
    """
    take the path of an image with apples, use the parameters provided
    and return the number of apples found
    parameters = [tolerance, i_erosion, i_closing, s_kernel]
    tolerance between 0 and 255 for the magic wand
    i_erosion : number of iteration of erosion
    i_closing : number of iteration of closing
    i_dilation : number of iteration of dilation
    s_kernal : width and height of the square kernel
    """
    tolerance = parameters[0]
    i_erosion = parameters[1]
    i_closing = parameters[2]
    i_dilation = parameters[3]
    s_kernel = parameters[4]


    src = cv.imread(source_path)
    src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    
    yellow_apple_color_hsv_deg_and_percent = [58, 75, 100]
    yellow_apple_hsv_255 = hsv_degre_and_percent_to_255(yellow_apple_color_hsv_deg_and_percent)

    mask_yellow_apple = magic_wand(src_hsv,yellow_apple_hsv_255,tolerance)

    kernel = np.ones((s_kernel,s_kernel),np.uint8)
    # closing and erosion
    erosion = cv.erode(mask_yellow_apple,kernel,iterations = i_erosion)
    closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel, 
                                iterations=i_closing)
    dilation = cv.dilate(closing, kernel, iterations=i_dilation)

    dilationu8 = np.uint8(dilation)

    Components_Stats = cv.connectedComponentsWithStats(dilationu8)

    if show:
        plt.figure(figsize = (10,10))
        plt.title("Our mask")
        plt.imshow(dilation,cmap= "gray")
        plt.show()

    return Components_Stats,dilation



# %%

path_imgs = "MinneApple/imgs/detection/train/images/"
path_masks = "MinneApple/imgs/detection/train/masks/"


n_img = 50

onlyfiles = [f for f in listdir(path_imgs) if isfile(join(path_imgs, f))]

trunc_file_names = onlyfiles[0:n_img]
parameters = [60,2,5,1,3]
L_truth,L_found,L_true_masks,L_found_masks,L_names = [],[],[],[],[]

for file_name in trunc_file_names:
    img_path = path_imgs + file_name
    mask_path = path_masks + file_name
    

    truth,true_mask = ground_truth(mask_path,show = False)
    found,found_mask = count_apples(img_path,parameters,show = False)
    print(parameters,file_name)
    print("truth:",truth[0],"found:",found[0])
    L_truth.append(truth[0])
    L_found.append(found[0])
    L_names.append(file_name)
    L_found_masks.append(found_mask)
    L_true_masks.append(true_mask)


# %%
L_errors = [L_found[i]-L_truth[i] for i in range(len(L_truth))]
plt.title("error on the number of apples found over 50 pictures")
for i_point in range(len(L_errors)):
    plt.plot([i_point,i_point],[0,L_errors[i_point]],c="r")
plt.plot((0,n_img),(0,0),c="black")
#plt.ylim(-30,30)
plt.show()

print("mean error:",statistics.mean(L_errors))

print("standard deviation:",statistics.stdev(L_errors))

print("sum of errors:",sum(L_errors))

print("in total, my algorithm counted",sum(L_found),"apples on",n_img,"images, but it should have counted a total of",sum(L_truth),"apples instead")
# %%
plt.plot([0,0],[0,1],c="r")
plt.show()
# %%
