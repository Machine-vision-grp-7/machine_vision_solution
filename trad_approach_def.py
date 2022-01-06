import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
from os import listdir
from os.path import isfile, join
import statistics
import sys
from numpy.core.numeric import outer

path_imgs = "MinneApple/imgs/detection/train/images/"
path_masks = "MinneApple/imgs/detection/train/masks/"



def magic_wand(img_hsv,color,tolerance):
    lowerBound = np.array(list_dif(color,tolerance))
    upperBound = np.array(list_sum(color,tolerance))
    mask2 = False
    mask3 = False
    #used for circling around the HSV cylindrical color space
    if lowerBound[0]<0:
        lowerBound2 = lowerBound.copy()
        upperBound2 = upperBound.copy()
        upperBound2[0] = 360
        lowerBound2[0] = 360 + lowerBound[0]
        lowerBound[0] = 0
        mask2 = True
    if upperBound[0]>360:
        lowerBound3 = lowerBound.copy()
        upperBound3 = upperBound.copy()
        lowerBound3[0] = 0
        upperBound3[0] = upperBound[0]-360
        upperBound[0] = 360
        mask3 = True

    #generate the mask
    mask=cv.inRange(img_hsv,lowerBound,upperBound)

    #used to generate the additional pask if we circle around
    #the hsv color space
    if mask2:
        mask2=cv.inRange(img_hsv,lowerBound,upperBound)
        mask = mask + mask2
    if mask3:
        mask3=cv.inRange(img_hsv,lowerBound,upperBound)
        mask = mask + mask3
    return mask
    

def list_dif(a,b):
    """
    calculate the difference between each item of 2 lists
    """
    return [a_i - b_i for a_i, b_i in zip(a, b)]

def list_sum(a,b):
    """
    calculate the sum between each item of 2 lists
    """
    return [a_i + b_i for a_i, b_i in zip(a, b)]


def hsv_degre_and_percent_to_255(hsv_degre_and_percent):
    """convert an HSV color with ranges [0-360,0-100,0-100]
    to [0-180,0-255,0-255]"""
    hsv_255 = [0,0,0]
    hsv_255[0] = hsv_degre_and_percent[0]*180/360
    hsv_255[1] = hsv_degre_and_percent[1]*255/100
    hsv_255[2] = hsv_degre_and_percent[2]*255/100
    return hsv_255


def hsv_255_to_degre_and_percent(hsv_255):
    """convert an HSV color with ranges [0-180,0-255,0-255]
    to [0-360,0-100,0-100]"""
    hsv_degre_and_percent = [0,0,0]
    hsv_degre_and_percent[0] = hsv_255[0]*360/180
    hsv_degre_and_percent[1] = hsv_255[1]*100/255
    hsv_degre_and_percent[2] = hsv_255[2]*100/255
    return hsv_degre_and_percent

def model_of_nb_of_apple(x):
    A = 1.5-0.1*(x-6)**2
    B = 0.4-0.001*(x-40)**2
    if x>80:
        return 0.4-0.001*(80-40)**2
    return max([A,B])


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

def generate_cool_mask(input_mask):
    components = cv.connectedComponentsWithStats(input_mask)
    output_mask = input_mask.copy()
    for i in range(len(components[3])): #centroids
        X,Y = components[3][i][0],components[3][i][1]
                # Center coordinates
        center_coordinates = (int(X),int(Y))
        
        # Radius of circle
        radius = 20
        
        # Blue color in BGR
        color = 255
        
        # Line thickness of 2 px
        thickness = 4
        
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        output_mask = cv.circle(output_mask, center_coordinates, radius, color, thickness)
    return output_mask

def draw_on_original(input_mask,img_path):
    modified_image = cv.imread(img_path)
    modified_image = np.array(modified_image[:,:,::-1])
    components = cv.connectedComponentsWithStats(input_mask)
    output_mask = input_mask.copy()
    for i in range(len(components[3])): #centroids
        X,Y = components[3][i][0],components[3][i][1]
                # Center coordinates
        center_coordinates = (int(X),int(Y))
        
        # Radius of circle
        radius = 20
        
        # Blue color in BGR
        color = (255,0,0)
        
        # Line thickness of 2 px
        thickness = 4
        
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        modified_image = cv.circle(modified_image, center_coordinates, radius, color, thickness)
    return modified_image

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
    i_opening = parameters[1]
    s_kernel = parameters[2]

    src = cv.imread(source_path)
    src_hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    
    color_hsv_255 = hsv_degre_and_percent_to_255(color_hsv_deg_and_percent)

    mask_apple = magic_wand(src_hsv,color_hsv_255,tolerance)
    
    #kernel = np.ones((s_kernel,s_kernel),np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(s_kernel,s_kernel))
    # erosion then dilation = closing
    mask_apple = cv.morphologyEx(mask_apple, cv.MORPH_OPEN, kernel,iterations=i_opening)

    return mask_apple


def count_apples(source_path,parameters=[[30,30,30],2,3],show = True,tech = "mean",opening_yellow = 0):
    """
    take the path of an image with apples, use the parameters provided
    and return the number of apples found
    parameters = [[tolerance_H,tolerance_S,tolerance_V], i_opening, s_kernel]
    [tolerance_H,tolerance_S,tolerance_V] between 0 and 255 for the magic wand
    i_opening : nb of opening
    s_kernal : width and height of the square kernel
    """
    red_color_hsv_deg_and_percent = [355, 73, 89] #apple_color_hsv_deg_and_percent
    yellow_hsv_deg_and_percent = [58, 75, 100] #apple_color_hsv_deg_and_percent
    yellow_param = parameters.copy()
    yellow_param[1]+= opening_yellow
    dilation_yellow = create_mask_with_color(source_path,yellow_param,yellow_hsv_deg_and_percent)
    dilation_red = create_mask_with_color(source_path,parameters,red_color_hsv_deg_and_percent)

    dilationu8_yellow = np.uint8(dilation_yellow)
    dilationu8_red = np.uint8(dilation_red)


    Components_Stats_yellow = cv.connectedComponentsWithStats(dilationu8_yellow)
    Components_Stats_red = cv.connectedComponentsWithStats(dilationu8_red)

    P_a = model_of_nb_of_apple(Components_Stats_yellow[0])
    P_b = model_of_nb_of_apple(Components_Stats_red[0])

    if show:
        plt.figure(figsize = (10,10))
        plt.title("Our yellow mask")
        plt.imshow(dilation_yellow,cmap= "gray")
        plt.show()
        plt.figure(figsize = (10,10))
        plt.title("Our red mask")
        plt.imshow(dilation_red,cmap= "gray")
        plt.show()
        if tech == "proba":
            print(P_a,P_b)
            if P_a > P_b:
                print("we keep the yellow mask")
            else:
                print("we keep the red mask")

    if tech == "sum":
        if Components_Stats_yellow[0]>100 and Components_Stats_red[0]>25:
            return Components_Stats_red,dilationu8_red
        else:
            final_mask = dilationu8_yellow + dilationu8_red
            Components_Stats_sum = cv.connectedComponentsWithStats(final_mask)
            return Components_Stats_sum,final_mask
    if tech == "mean":
        mean = [0,0]
        mean[0] = (Components_Stats_yellow[0] + Components_Stats_red[0])/2
        return mean,dilationu8_yellow
    elif tech == "proba":    
        if P_a > P_b:
            return Components_Stats_yellow,dilationu8_yellow
        else:
            return Components_Stats_red,dilationu8_red


    


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
    return L_true_masks,L_truth,L_names,L_etiquette,L_height

def do_n_img(n,parameters,opening_i):
    path_imgs = "MinneApple/imgs/detection/train/images/"
    path_masks = "MinneApple/imgs/detection/train/masks/"


    n_img = n

    onlyfiles = [f for f in listdir(path_imgs) if isfile(join(path_imgs, f))]

    trunc_file_names = onlyfiles[0:n_img]
     #et 1 1 pour ero et dila yellow
    L_truth,L_found,L_true_masks,L_found_masks,L_names = [],[],[],[],[]
    i=0
    for file_name in trunc_file_names:
        i+=1
        img_path = path_imgs + file_name
        mask_path = path_masks + file_name
        

        truth,true_mask = ground_truth(mask_path,show = False)
        found,found_mask = count_apples(img_path,parameters,show = False,tech = "sum",opening_yellow=opening_i)
        #print(parameters,file_name)
        #print("truth:",truth[0],"found:",found[0])
        L_truth.append(truth[0])
        L_found.append(found[0])
        L_names.append(file_name)
        L_found_masks.append(found_mask)
        L_true_masks.append(true_mask)
        percent = "#"*int(50*i/n_img) + "-" * (50-int(50*i/n_img)) + " "+str(round(100*i/n_img,2)) + "%   "
        sys.stdout.write("\r"+ percent)
        sys.stdout.flush()

    L_errors = [L_found[i]-L_truth[i] for i in range(len(L_truth))]
    plt.title("error on the number of apples found over " + str(n_img)+" pictures")
    for i_point in range(len(L_errors)):
        plt.plot([i_point,i_point],[0,L_errors[i_point]],c="r")
    plt.plot((0,n_img),(0,0),c="black")
    #plt.ylim(-30,30)
    plt.show()

    print("mean error:",statistics.mean(L_errors))

    print("sum of errors:",sum(L_errors))

    print("In total, my algorithm counted",sum(L_found),"apples on",
    n_img,"images, but it should have counted a total of",
    sum(L_truth),"apples instead.")
    print("This is an error rate of", round(100*abs(sum(L_errors)/sum(L_truth)),2),"%.")
    SMAE = 0
    SMSE = 0
    y_barre = statistics.mean(L_truth)
    S_yi_minus_ychapi_squared = 0
    S_yi_minus_ybarre_squared = 0
    for i in range(len(L_errors)):
        SMAE += abs(L_errors[i])
        SMSE += L_errors[i]**2
        S_yi_minus_ychapi_squared+= (L_truth[i]-L_found[i])**2
        S_yi_minus_ybarre_squared += (L_truth[i]-y_barre)**2
    MAE= (1/n_img) * SMAE
    RSME= ((1/n_img) * SMSE)**0.5
    R2 = 1-(S_yi_minus_ychapi_squared/S_yi_minus_ybarre_squared)
    print("MAE =",MAE)
    print("RSME =",RSME)
    print("R2 =",R2)
    dif = sum(L_truth)-sum(L_found)
    return (sum(L_truth),dif,MAE,RSME,R2)