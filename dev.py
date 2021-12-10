#%%
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


img = cv2.imread('tree_petit.jpg')
print(img.shape)


img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img_only_red = img[:,:,2]


plt.imshow(img_only_red,cmap = "gray")
plt.show()

thresh2, only_red_thresd = cv2.threshold(img_only_red, thresh=127, maxval=255, type=cv2.THRESH_OTSU)  # otsu's method

plt.imshow(only_red_thresd,cmap = "gray")
plt.show()

# %%

def keep_color_px(img_color, desired_color):
    threshold = 230
    for i in range(len(img_color)):
        for j in range(len(img_color[0])):
                px_color = img_color[i][j] 
                dist = np.linalg.norm(px_color-desired_color)
                if dist>threshold:
                    img_color[i][j] = 255
                else: 
                    img_color[i][j] = 0

    return img_color


img = cv2.imread('tree_petit.jpg')
img = img[:,:,::-1] #RGB -> BGR

plt.imshow(img)
plt.show()

img_kept_color = keep_color_px(img,[93,102,253])
plt.imshow(img_kept_color)
plt.show()
# %%
