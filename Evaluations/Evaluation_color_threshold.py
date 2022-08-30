import numpy as np
import os
import cv2
import imutils


def mean(image):
    b,g,r = cv2.split(image)
    mean_r = int(np.mean(r))
    mean_g = int(np.mean(g))
    mean_b = int(np.mean(b))

    if (mean_r >= mean_g) and (mean_r >= mean_b):
        return "R",mean_r,mean_g,mean_b
    if (mean_g >= mean_b) and (mean_g >= mean_r):
        return "G",mean_r,mean_g,mean_b
    if (mean_b >= mean_g) and (mean_b >= mean_r):
        return "B",mean_r,mean_g,mean_b

path = "../Colors"
for i in os.listdir(path):
    size_patch = 32
    image = cv2.imread(os.path.join(path,i))

    color_main, color_mean_r, color_mean_g, color_mean_b = mean(image)

    x,y,z = image.shape
    n_var = ((size_patch**2)*3)+3
    last_ = (size_patch**2)*3

    patch_temp = np.reshape([0 for _ in range(last_)],(size_patch,size_patch,3)).astype(np.uint8)
    x_max,y_max,_ = imutils.rotate_bound(patch_temp,45).shape

    add_main = None
    if color_main == "R":
        channel_c = 3
        add_main = min(color_mean_r,max(color_mean_b,color_mean_g)+ 50)
    if color_main == "G":
        channel_c = 2
        add_main = min(color_mean_g,max(color_mean_b,color_mean_r)+ 50)
    if color_main == "B":
        channel_c = 1
        add_main = min(color_mean_b,max(color_mean_r,color_mean_g)+ 50)


    x_l = [add_main if (((i+1)%3)-channel_c)%3==0 else 0 for i in range(n_var)]
    x_l[-1] = 0
    x_l[-2] = 0
    x_l[-3] = 0
    x_u = []
    for i in range(n_var):
        if i == (last_):
            x_u.append(x-x_max)
        if i == (last_+1):
            x_u.append(y-y_max)
        if i == (last_+2):
            x_u.append(360)
        if (i < last_):
            if (((i+1)%3)-channel_c)%3==0:
                x_u.append(255)
            else:
                if (((i+1)%3)-1)%3==0:
                    x_u.append(color_mean_b+20)
                if (((i+1)%3)-2)%3==0:
                    x_u.append(color_mean_g+20)
                if (((i+1)%3)-3)%3==0:
                    x_u.append(color_mean_r+20)

