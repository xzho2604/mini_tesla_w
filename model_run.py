import cv2
import numpy as np
import glob
import sys
import time

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

def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


model_path = 'my_model.h5'

#model = model_from_json(open(modeli_path).read())
#model.load_weights(os.path.join(os.path.dirname(modelFile), 'model_weights.h5'))
model = load_model(model_path, custom_objects={'multitask_loss': multitask_loss})
print("model loaded")
