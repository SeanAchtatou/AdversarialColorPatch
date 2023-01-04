from art.attacks.evasion import SquareAttack, AdversarialPatch, SaliencyMapMethod, CarliniLInfMethod
from art.estimators.classification import KerasClassifier
from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
import time
import os



dir_Results = "Attacks_Results"

tf.compat.v1.disable_eager_execution()
model_ = keras.models.load_model("Models/Road_Signs_classifier_model.h5")

imageS = np.expand_dims(cv2.resize(cv2.imread("images_/stop_sign_close.jpg").astype(np.float32),(30,30)),0)
imageR = np.expand_dims(cv2.resize(cv2.imread("images_/r_4.jpg").astype(np.float32),(30,30)),0)
image3 = np.expand_dims(cv2.resize(cv2.imread("images_/30_2.jpg").astype(np.float32),(30,30)),0)

images = [imageS,imageR,image3]
imagesNames = ["Stop","Round","Speed"]

clip_values = (0, 255)
nb_classes  = 42
batch_size = 16
scale_min = 0.1
scale_max = 1.0
rotation_max = 180
learning_rate = 5000.
max_iter = 500

model = KerasClassifier(model=model_,clip_values=(0,255))

ss = np.array([[0 for i in range(0,43)]])
ss[0][3] = 1


start = None
end = None
def attack(attack,im):
    return attack.generate(x=im,y=ss)

if "__main__" == __name__:

    #SA = SquareAttack(estimator=model,eps=5,max_iter=1000,nb_restarts=10,batch_size=batch_size)
    #AT = AdversarialTexturePyTorch(model) #More to define
    AP = AdversarialPatch(classifier=model, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max,
                          learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size)

    SP = SaliencyMapMethod(classifier=model,batch_size=batch_size)

    CP = CarliniLInfMethod(classifier=model, batch_size=batch_size)


    attacks = [SP,CP]
    attackUN = False

    pos = -1
    for im in images:
        pos +=1
        for i in attacks:
            start = time.time()
            adv_patch = attack(i,im)
            end = time.time()
            print(f"Time for the attack: {end-start}")

            if i == AP:
                working_size = []
                mask = adv_patch[1]
                adv_patch_mask = adv_patch[0]
                adv_patch_ = (adv_patch_mask * mask).astype(np.uint8)
                for size in np.arange(0.1,0.6,0.1):
                    count = 10
                    while True:
                        #cv2.imshow("The Patch Applied",adv_patch_)
                        #cv2.waitKey(1)
                        patches_image = (AP.apply_patch(im,scale=size)).astype(np.uint8)
                        show_p_i = np.squeeze(patches_image,0)
                        #cv2.imshow("The Patch Applied",show_p_i)
                        #cv2.waitKey(0)

                        if (patches_image == im).all():
                            print("Error, same image as before.")



                        else:
                            x = model_.predict(patches_image).argmax(1)
                            xp = model_.predict(im).argmax(1)
                            print("Prediction without patch:",xp)
                            print("Prediction with patch:",x)
                            adv_patch = np.squeeze(patches_image,0).astype(np.uint8)
                            adv_patch = cv2.resize(adv_patch,(100,100))
                            cv2.imshow("Final Image",adv_patch)
                            cv2.waitKey(1)
                            count -= 1
                            if (xp != x) or (count == 0):
                                if count == 0:
                                    print(f"Can't work with {size}")
                                else:
                                    print(f"Work with {size}")
                                    working_size.append(size)

                                    cv2.imwrite(f"{dir_Results}/AP/{imagesNames[pos]}image_s{size}_P_{x}.png",adv_patch)
                                    cv2.imwrite(f"{dir_Results}/AP/{imagesNames[pos]}patch_s{size}_P_{x}.png",adv_patch_)
                                break

                print(f"Work with all these sizes : {working_size}")

            if i == SP:
                if (adv_patch == im).all():
                    print("Error, same image as before.")


                else:

                    x = model_.predict(adv_patch).argmax(1)
                    xp = model_.predict(im).argmax(1)
                    print("Prediction without patch:",xp)
                    print("Prediction with patch:",x)
                    adv_patch = np.squeeze(adv_patch,0).astype(np.uint8)
                    adv_patch = cv2.resize(adv_patch,(100,100))
                    cv2.imshow("Final Image",adv_patch)
                    cv2.waitKey(1)
                    cv2.imwrite(f"{dir_Results}/SP/{imagesNames[pos]}image_P_{x}.png",adv_patch)


            if i == CP:
                if (adv_patch == im).all():
                    print("Error, same image as before.")


                else:
                    x = model_.predict(adv_patch).argmax(1)
                    xp = model_.predict(im).argmax(1)
                    print("Prediction without patch:",xp)
                    print("Prediction with patch:",x)
                    adv_patch = np.squeeze(adv_patch,0).astype(np.uint8)
                    adv_patch = cv2.resize(adv_patch,(100,100))
                    cv2.imshow("Final Image",adv_patch)
                    cv2.waitKey(1)
                    cv2.imwrite(f"{dir_Results}/CP/{imagesNames[pos]}image_P_{x}.png",adv_patch)




