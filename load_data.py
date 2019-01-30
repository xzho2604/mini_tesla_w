#read all the image file with label and create a np file with train=X and train_label =y as data.npz
import cv2
import numpy as np
import glob
import sys
import time
import os
import re

#path to the data
path = './*'
img_shape = (120,240)
#===================================================================================
def load_data(img_shape, path):
    print("loading training data")
    
    #init the empty np array to read trainig data from file to memmory
    #image shape is (120,240) 
    X = np.empty((0,img_shape[0],img_shape[1],1))
    y = np.empty((0, 5))

    training_data = glob.glob(path) #list of './1691_10000.jpg'
    if not training_data:
        print("Data not found, exit")
        sys.exit()

    #loop through the training data the stack image into the X and label into y
    count = 0 #use to indicae the process of loading
    for img_path in training_data:
        #extract the laebel np vector from the image path name
        label_str = re.search(r'_([0-9]+)', img_path)
        if label_str != None: #only parese valid file
            label_str = label_str.group(1)
            label = np.array([int(x) for x in list(label_str)]).reshape(1,-1) #label of dim 1,5:[[1 0 0 1 0]]
            y = np.vstack((y,label)) #stack the label vertically

            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE).reshape(1,img_shape[0],img_shape[1],1) #read grey image into np vectors
            X = np.vstack((X,img)) #stack vertically each training example

        print(count)
        count += 1

    #now we load all the training data in X and label in y
    print("X has shape:",X.shape)
    print("y has shape:",y.shape)

    #now we have X as original set and y as label
    #save X and y as np file
    np.savez('data.npz',train = X,train_label = y)
    print("file saved!")


#===================================================================================
def main():
    load_data(img_shape,path)

if __name__ == "__main__":
    main()
