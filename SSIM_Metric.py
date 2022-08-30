import sewar
import cv2
import os
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import time

SSIM = sewar.full_ref.ssim
#VIFP = sewar.full_ref.vifp
SCC = sewar.full_ref.scc
#UQI = sewar.full_ref.uqi
MSE = sewar.full_ref.mse
MSSSIM = sewar.full_ref.msssim

model = keras.models.load_model("Models/Road_Signs_classifier_model.h5")
#model = keras.models.load_model("Models/Road_Signs_classifier_model_ResNet_1000.h5")

def simulate(o,im):
    ssim_values = []
    scc_values = []
    msssim_values = []
    uqi_values = []
    distance = min(im.shape[0],im.shape[1]) - 30
    x = im.shape[0] - distance
    y = im.shape[1] - distance
    for i in range(1,distance):
        im_ = cv2.resize(im,(y+i,x+i)).astype(np.uint8)
        o_ = cv2.resize(o,(y+i,x+i)).astype(np.uint8)
        cv2.imshow("Image",im_)
        cv2.waitKey(1)
        im_model = cv2.resize(im_,(30,30))
        im_model = np.expand_dims(im_model,0)
        o_model = cv2.resize(o_,(30,30))
        o_model = np.expand_dims(o_model,0)
        r = model.predict(im_model).argmax(1)
        s = model.predict(o_model).argmax(1)
        if (r == s):
            print("     Original sign detected!")
            ssim_values.append(0)
        else:
            v = similarity_SSIM(o_,im_)
            ssim_values.append(v[1])
            v = similarity_SCC(o_,im_)
            scc_values.append(v)
            v = similarity_MSSSIM(o_,im_)
            msssim_values.append(v)
            #v = similarity_UQI(o_,im_)
            #uqi_values.append(v)
            similarity_MSE(o_,im_)


    plt.xlim(1,distance)
    plt.ylim(0,1)
    plt.xlabel('Object Size')
    plt.ylabel('Similarities Percentage')
    plt.plot(ssim_values,"ko",label=f"SSIM",markersize=1)
    plt.plot(scc_values,"bo",label=f"SCC",markersize=1)
    plt.plot(uqi_values,"ro",label=f"UQI",markersize=1)
    plt.legend(loc="upper center",fontsize=5)

    plt.show()


def similarity_MSSSIM(a,b):
    comp = MSE(a,b)
    print(f"        MSSSIM:{comp}")
    return comp

def similarity_MSE(a,b):
    comp = MSE(a,b)
    print(f"        MSE:{comp}")
    return comp

def similarity_SCC(a,b):
    comp = SCC(a,b)
    print(f"        SCC:{comp}")
    return comp

def similarity_UQI(a,b):
    comp = UQI(a,b)
    print(f"        UQI:{comp}")
    return comp

def similarity_VIFP(a,b):
    comp = VIFP(a,b)
    print(f"        VIFP:{comp}")
    return comp

def similarity_SSIM(a,b):
    comp = SSIM(a,b)
    print(f"        SSIM:{comp[1]}")
    return comp

if __name__ == "__main__":
    stop = True
    speed = False
    images = []
    image = None
    if stop:
        image = cv2.imread("images_/stop13.jpg")
        image1 = "ExperimentsMultiplesSigns/Stop_4/Images/patch_image_102S_487X457Y_344A_1660767012T.png"
        image2 = "ExperimentsMultiplesSigns/Stop_4/Images/patch_image_306S_369X187Y_117A_1660770730T.png"
        image3 = "images_/stop13.jpg"
        imim = [image3]
        for k in imim:
            images.append(cv2.imread(k))

    if speed:
        image = cv2.imread("images_/30_km_traffic_sign.jpg")
        image1 = "patches/p_image/patch_image_48S_167X74Y_305A_1654729044T.png"
        image2 = "patches/p_image/patch_image_54S_211X82Y_82A_1654711840T.png"
        image3 = "patches/p_image/patch_image_110S_172X67Y_139A_1654766037T.png"
        im2 = cv2.imread(image1)
        im3 = cv2.imread(image2)
        im4 = cv2.imread(image3)
        images += [im2,im3,im4]


    print("Simple similarities SSIM with original images:")
    for i in images:
        _ = similarity_SSIM(image,i)
        _ = similarity_SCC(image,i)

    #print("Simulation with SSIM:")
    #for i in images:
        #simulate(image,i)



