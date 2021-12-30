#do a "pip install patchify" beforehand!!
import patchify
import numpy as np
import cv2 as cv

def cut_in_patches(path_to_picture,patch_size):
    """This function cuts an image into multiple
    patches and return an array "output" which can be
    used for apple counting.
    path_to_picture : (str) path to the picture
    patch_size : (int) desired size of the patches in pixels
    
    to use the function, do :

        X = cut_in_patches(img_path,180)

        X.shape gives (28, 180, 180, 3)"""
    image = cv.imread(path_to_picture) #import image as an array
    image = image[:,:,::-1]  #convert to RGB
    
    patches = patchify.patchify(image, (patch_size,patch_size,3), step=patch_size)

    output = []

    for i in range(len(patches)):
        for j in range(len(patches[0])):
            output.append(patches[i][j][0]) #create the output array (X)

    return np.asarray(output)

