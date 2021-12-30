import cv2 as cv
import numpy as np

def nb_of_apples(mask_path):
    true_mask_src = cv.imread(mask_path)
    true_mask_gray = cv.cvtColor(true_mask_src, cv.COLOR_BGR2GRAY)
    true_mask_srcu8 = np.uint8(true_mask_gray)
    Components_Stats = cv.connectedComponentsWithStats(true_mask_srcu8)
    #Components_Stats[0] is the number of apples + 1
    return Components_Stats[0]