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
                if k != 0:
                    if abs(img_hsv[i][j][k]-color[k]) > tolerance[k]:
                        keep = False
                else:
                    R_lim_angle = color[0] + tolerance[k]
                    L_lim_angle = color[0] - tolerance[k]

                    if (abs(img_hsv[i][j][0]-color[0]) > tolerance[k]):

                        if L_lim_angle < 0:
                            if (abs((img_hsv[i][j][0]-255)-color[0]) > tolerance[k]):
                                keep = False
                        elif R_lim_angle > 255:
                            if abs((img_hsv[i][j][0]+255)-color[0]) > tolerance[k]:
                                keep = False
                        else:
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


def show_img(path_or_img):
    
    if type(path_or_img) == str:
        image = cv.imread(path_or_img)
        image_conv = image[:,:,::-1]
        plt.figure(figsize = (10,10))
        plt.axis('off')
        plt.title("Image")
        plt.imshow(image_conv)
        plt.show()
    elif type(path_or_img[0][0]) != list:
        plt.figure(figsize = (10,10))
        plt.axis('off')
        plt.title("Image")
        plt.imshow(path_or_img,cmap="gray")
        plt.show()
    else:
        plt.figure(figsize = (10,10))
        plt.axis('off')
        plt.title("Image")
        plt.imshow(path_or_img[:,:,::-1])
        plt.show()


def ground_truth(file_path,show = True,verbose=1):
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

def create_mask_with_color(source_path,parameters,color_hsv_deg_and_percent):
    
    tolerance = parameters[0]
    i_erosion = parameters[1]
    i_dilation = parameters[2]
    s_kernel = parameters[3]

    src = cv.imread(source_path)
    src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    
    color_hsv_255 = hsv_degre_and_percent_to_255(color_hsv_deg_and_percent)

    mask_apple = magic_wand(src_hsv,color_hsv_255,tolerance)
    
    kernel = np.ones((s_kernel,s_kernel),np.uint8)
    # erosion then dilation = closing
    erosion = cv.erode(mask_apple,kernel,iterations = i_erosion)
    dilation = cv.dilate(erosion, kernel, iterations=i_dilation)

    erosion = cv.erode(dilation,kernel,iterations = i_erosion)
    dilation = cv.dilate(erosion, kernel, iterations=i_dilation)
    return dilation


def count_apples(source_path,parameters=[[30,30,30],2,2,3],show = True):
    """
    take the path of an image with apples, use the parameters provided
    and return the number of apples found
    parameters = [[tolerance_H,tolerance_S,tolerance_V], i_erosion, i_closing, s_kernel]
    [tolerance_H,tolerance_S,tolerance_V] between 0 and 255 for the magic wand
    i_erosion : number of iteration of erosion
    i_dilation : number of iteration of dilation
    s_kernal : width and height of the square kernel
    """
    red_color_hsv_deg_and_percent = [358, 68, 88] #apple_color_hsv_deg_and_percent
    yellow_hsv_deg_and_percent = [58, 75, 100] #apple_color_hsv_deg_and_percent

    dilation_yellow = create_mask_with_color(source_path,parameters,yellow_hsv_deg_and_percent)
    dilation_red = create_mask_with_color(source_path,parameters,red_color_hsv_deg_and_percent)

    dilationu8_yellow = np.uint8(dilation_yellow)
    dilationu8_red = np.uint8(dilation_red)


    Components_Stats_yellow = cv.connectedComponentsWithStats(dilationu8_yellow)
    Components_Stats_red = cv.connectedComponentsWithStats(dilationu8_red)


    if show:
        plt.figure(figsize = (10,10))
        plt.title("Our yellow mask")
        plt.imshow(dilation_yellow,cmap= "gray")
        plt.show()
        plt.figure(figsize = (10,10))
        plt.title("Our red mask")
        plt.imshow(dilation_red,cmap= "gray")
        plt.show()

    return Components_Stats_yellow,dilationu8_yellow,Components_Stats_red,dilationu8_red

def ground_truth_for_all_data(verbose=0):
    
    path_masks = "MinneApple/imgs/detection/train/masks/"

    onlyfiles = [f for f in listdir(path_imgs) if isfile(join(path_imgs, f))]

    trunc_file_names = onlyfiles
    L_truth,L_true_masks,L_names = [],[],[]

    for file_name in trunc_file_names:
        mask_path = path_masks + file_name
        

        truth,true_mask = ground_truth(mask_path,show = False,verbose=0)
        if verbose == 1:
            print(file_name)
            print("truth:",truth[0])
        L_truth.append(truth[0])
        L_names.append(file_name)
        L_true_masks.append(true_mask)

    from collections import Counter

    L_compte_truth = Counter(L_truth).most_common()
    if verbose == 1:
        print(L_compte_truth)
    plt.title("Y odccurences of pictures with X number of apples")
    L_height = list(zip(*L_compte_truth))[1]
    L_etiquette = list(zip(*L_compte_truth))[0] 
    plt.bar(L_etiquette,L_height)
    plt.show()
    return L_true_masks,L_truth,L_names

# %%

path_imgs = "MinneApple/imgs/detection/train/images/"
path_masks = "MinneApple/imgs/detection/train/masks/"


n_img = 50

onlyfiles = [f for f in listdir(path_imgs) if isfile(join(path_imgs, f))]

trunc_file_names = onlyfiles[0:n_img]
parameters = [[30,30,30],2,2,3]
L_truth,L_found,L_true_masks,L_found_masks,L_names = [],[],[],[],[]
# %%
for file_name in trunc_file_names:
    img_path = path_imgs + file_name
    mask_path = path_masks + file_name
    

    truth,true_mask = ground_truth(mask_path,show = False)
    found_yellow,found_yellow_mask,found_red,found_red_mask = count_apples(img_path,parameters,show = False)
    print(parameters,file_name)
    print("truth:",truth[0],"found:",found_yellow[0])
    L_truth.append(truth[0])
    L_found.append(found_yellow[0])
    L_names.append(file_name)
    L_found_masks.append(found_yellow_mask)
    L_true_masks.append(true_mask)


# %%
L_errors = [L_found[i]-L_truth[i] for i in range(len(L_truth))]
plt.title("error on the number of apples found over " + str(n_img)+" pictures")
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
# %%

path_imgs = "MinneApple/imgs/detection/train/images/"
path_masks = "MinneApple/imgs/detection/train/masks/"


n_img = 50

onlyfiles = [f for f in listdir(path_imgs) if isfile(join(path_imgs, f))]

trunc_file_names = onlyfiles[0:n_img]
parameters = [[30,30,30],2,2,3]
L_truth,L_found,L_true_masks,L_found_masks,L_names = [],[],[],[],[]

for file_name in trunc_file_names:
    img_path = path_imgs + file_name
    mask_path = path_masks + file_name
# %%    
parameters = [[20,80,80],2,0,3]
rouge = "20150921_131346_image211.png"
verte_chelou = "20150919_174730_image1331.png"
yellow_green = "20150919_174151_image1.png"

img_path = path_imgs + yellow_green
mask_path = path_masks + yellow_green

show_img(img_path)

show_img(mask_path)

L_results = []
i = 0
for param1 in [15,20,25,30,35]:
    for param2_3 in [60,80,100]:
        for param4 in [0,1,2]:
            for param5 in [0,1,2]:
                parameters = [[param1,param2_3,param2_3],param4,param5,3]
                found_yellow,found_yellow_mask,found_red,found_red_mask = \
                count_apples(img_path,parameters,show = False)
                dist = 71 - found_yellow[0]
                L_results.append((parameters,dist))
                print(i)
print(L_results)
#%%
truth,true_mask = ground_truth(mask_path,show = False)
found_yellow,found_yellow_mask,found_red,found_red_mask = \
     count_apples(img_path,[[20, 100, 100], 2, 0, 3],show = False)
print(truth[0],found_yellow[0],found_red[0])
show_img(found_yellow_mask)
show_img(found_red_mask)


#%%
L_results.sort(key = lambda x: x[1])



Best = [x for x in L_results if abs(x[1])< 10]
















# %%
L_true_masks, L_truth, L_names = ground_truth_for_all_data()

