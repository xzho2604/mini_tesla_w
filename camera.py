import cv2
import numpy as np
import os
import keras.backend as K
from keras.models import Sequential
from keras import layers
from keras.layers import Dropout, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.models import load_model

#define global var
img_shape = (120,240) #(height, width)
#=================================================================================
#custom loss function for multi tasking
def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

#=================================================================================

#start capturing connectiong withe MJPEG server
#cap = cv2.VideoCapture('http://192.168.1.12:8080/?action=stream')

#load the previously trained model
model_path = 'my_model.h5'
model = load_model(model_path, custom_objects={'multitask_loss': multitask_loss})
print("Model Loaded!")


#test image
grey = cv2.imread('52_10000.jpg',0)
cv2.imshow('good',grey)
grey = grey.reshape(1,120,240,1)
grey = grey/255.0
print(model.predict(grey))
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
while True:
    _,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #change BRG to Grey
    cv2.imshow('Picam',grey) 

    #grey of shape (120,240) reshape the grey to (1,120,240,1)
    grey = grey.reshape(1,img_shape[0], img_shape[1],1)
    grey = grey / 255.0 #normalise the input image

    #use model to predict the control command
    prediction = model.predict(grey) #[[0.93078536 0.00600131 0.16780938 0.46101537 0.00867568]] 
    print(prediction)












    #if ESC pressed close the window
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
'''
