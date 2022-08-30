import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model




image = cv2.imread("images_/roundaboutclose.jpg")
image = np.expand_dims(cv2.resize(image,(50,50)),0)

model = tf.keras.models.load_model("Models/Road_Signs_classifier_model.h5")
#print(model.summary())
#config = model.get_config()
#x = config["layers"][0]["config"]["batch_input_shape"][1:3]
#print(x)
x = model.predict(image)

print(np.argmax(x))
print(x[0][np.argmax(x)])



