import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os

import cv2

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D


models_dir = "Models"

classes_ = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
            10:'No passing',
            11:'No passing veh over 3.5 tons',
            12:'Right-of-way at intersection',
            13:'Priority road',
            14:'Yield',
            15:'Stop',
            16:'No vehicles',
            17:'Veh > 3.5 tons prohibited',
            18:'No entry',
            19:'General caution',
            20:'Dangerous curve left',
            21:'Dangerous curve right',
            22:'Double curve',
            23:'Bumpy road',
            24:'Slippery road',
            25:'Road narrows on the right',
            26:'Road work',
            27:'Traffic signals',
            28:'Pedestrians',
            29:'Children crossing',
            30:'Bicycles crossing',
            31:'Beware of ice/snow',
            32:'Wild animals crossing',
            33:'End speed + passing limits',
            34:'Turn right ahead',
            35:'Turn left ahead',
            36:'Ahead only',
            37:'Go straight or right',
            38:'Go straight or left',
            39:'Keep right',
            40:'Keep left',
            41:'Roundabout mandatory',
            42:'End of no passing',
            43:'End no passing vehicle with a weight greater than 3.5 tons' }

def model_train():
    input_shape = (32,32,3)
    data = []
    labels = []
    classes = 43
    path = os.path.join(os.getcwd(),"archive","Train")
    for i in range(classes):
        path_ = os.path.join(path,str(i))
        images = os.listdir(path_)
        for j in images:
            try:
                image = cv2.imread(f"{path_}/{j}")
                image = cv2.resize(image,(32,32))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    data = np.array(data)
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=42)

    y_train = keras.utils.to_categorical(y_train,classes)
    y_test = keras.utils.to_categorical(y_test,classes)


    if False:
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(43, activation='softmax'))
        #Compilation of the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if False:
        base_model = VGG16(weights=None,include_top=True,input_shape=input_shape,classes=43)
        model = Sequential()
        model.add(base_model)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    if True:
        resnet = ResNet50(weights=None,include_top=False,input_shape=input_shape,classes=43)
        x = resnet.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(classes, activation= 'softmax')(x)
        model = Model(inputs = resnet.input, outputs = predictions)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    epoches = 50
    batchs = 32
    model.fit(x_train,y_train,batch_size=batchs, epochs=epoches,validation_data=(x_test,y_test))
    model.save(f"{models_dir}/Road_Signs_classifier_model_ResNet_Web.h5")
    print("Model saved!")

if "__main__" == __name__:
    model_train()