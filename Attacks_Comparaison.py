from art.attacks.evasion import SquareAttack, FastGradientMethod ,AdversarialTexturePyTorch, RobustDPatch, AdversarialPatch
from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np

tf.compat.v1.disable_eager_execution()
model_ = keras.models.load_model("Models/Road_Signs_classifier_model.h5")
image = np.expand_dims(cv2.resize(cv2.imread("images_/stop_sign_close.jpg").astype(np.float32),(30,30)),0)
#image = np.expand_dims(cv2.resize(cv2.imread("patches/p_image/patch_image_38S_190X178Y_280A_1656013696T.png").astype(np.float32),(30,30)),0)

ss = np.array([[0 for i in range(0,43)]])
ss[0][14] = 1

def attack(attack):
    patch = attack.generate(x=image,y=ss)
    return patch

if "__main__" == __name__:
    #model = TensorFlowV2Classifier(model=model_,input_shape=(30,30),nb_classes=42)
    model = KerasClassifier(model=model_,clip_values=(0,255))

    SA = SquareAttack(estimator=model,eps=5,max_iter=1000,nb_restarts=5)
    #AT = AdversarialTexturePyTorch(model) #More to define
    AP = AdversarialPatch(model)

    attacks = [SA,AP]

    for i in attacks:
        adv_patch = attack(i)
        if i == AP:
            mask = adv_patch[1].astype(np.uint8)
            #cv2.imshow("p",mask)
            #cv2.waitKey(0)
            adv_patch = adv_patch[0].astype(np.uint8)
            #cv2.imshow("p",adv_patch)
            #cv2.waitKey(0)
            w = mask == (0,0,0)
            image = np.squeeze(image,0).astype(np.uint8)
            #cv2.imshow("p",image)
            #cv2.waitKey(0)
            adv_patch[w] = image[w]
            #cv2.imshow("p",adv_patch)
            #cv2.waitKey(0)
            adv_patch = np.expand_dims(adv_patch,0)
            image = np.expand_dims(image,0)
        if (adv_patch == image).all():
            print("Error, same image as before.")
        x = model_.predict(adv_patch).argmax(1)
        xp = model_.predict(image).argmax(1)
        print("Prediction without patch:",xp)
        print("Prediction with patch:",x)
        adv_patch = np.squeeze(adv_patch,0).astype(np.uint8)
        adv_patch = cv2.resize(adv_patch,(100,100))
        cv2.imshow("P",adv_patch)
        cv2.waitKey(0)
