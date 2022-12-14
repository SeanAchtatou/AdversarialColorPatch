import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from PIL import Image
import os
import cv2

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, AveragePooling2D, Activation
from sklearn.model_selection import train_test_split
from tensorflow import keras

models_dir = "../Models"
names = ["Lenet_model","Alexnet_model","VGG16_model"]
chosen = 0

batch_size = 32

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

def VGG16_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3),  strides=(1,1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def Alexnet_model(input_shape):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model




def Lenet_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='tanh', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(64))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def training(x,y,test_x,test_y,model):
    epoches = 15
    batchs = 32
    model.fit(x,y,batch_size=batchs, epochs=epoches,validation_data=(test_x,test_y))
    model.save(f"{models_dir}/RoadSigns_classifier_{names[chosen]}.h5")
    print("Model saved!")



def data_load():
    global chosen
    models = [Lenet_model,Alexnet_model,VGG16_model]
    input_shape = (30,30,3)
    data = []
    labels = []
    classes = 43
    path = os.path.join(os.getcwd(), "../archive", "Train")
    for i in range(classes):
        path_ = os.path.join(path,str(i))
        images = os.listdir(path_)
        for j in images:
            try:
                image = cv2.imread(f"{path_}/{j}")
                image = cv2.resize(image,(30,30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    data = np.array(data)
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.4,random_state=42)

    y_train = keras.utils.to_categorical(y_train,classes)
    y_test = keras.utils.to_categorical(y_test,classes)

    count = 0
    print("Select the model to use:")
    for i in models:
        print(f"    [{count}]{i}")
        count += 1
    chosen = int(input(">"))
    model = models[chosen](input_shape)

    training(x_train,y_train,x_test,y_test,model)



if __name__ == "__main__":
    data_load()


