#import librairies  
import numpy as np
import random
import cv2
import os
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
    
#initiate parameters
learning_rate = 1e-2 #step size at each iteration 
batch_size = 32 #number of training in one iteration
epochs = 100 #number of passes of the entire training dataset
img_dims = (96,96,3) #specifying image dimensions

data = []
labels = []

#load images
image_files = [f for f in glob.glob(r'C:/Users/ethan/Desktop/Face recognition project/Gender' + "/**/*",
recursive=True) if not os.path.isdir(f)]

random.shuffle(image_files)

#converting images to arrays
for img in image_files:
    image = cv2.imread(img) #loads an image from the specified file
    image = cv2.resize(image, (img_dims[0],img_dims[1])) #resize image to dimensions 96/96
    image = img_to_array(image) #converts image in numpy array
    data.append(image)
    label = img.split(os.path.sep)[-2]
    #labeling genders
    if label == "woman":
        label = 1
    else:
        label = 0
    labels.append([label])

#transforms data into numpy array    
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

#splits array into random train and test subsets
x_train,x_test,y_train,y_test = train_test_split(data, labels, test_size=0.2,random_state=42)

#converts integers to binary class matrix
y_train = to_categorical(y_train, num_classes=2) #matrix 2x2
y_test = to_categorical(y_test, num_classes=2)

#data augmentation
#generate batches of tensor image data with real-time data augmentation
augmentation = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
                                  shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                  fill_mode="nearest")

#random rotation : performs random rotations
#width_shift_range : shift the image to the left or right
#height_shift_range : shift the image up or down
#shear_range : shear angle
#zoom_range : random zooms
#horizontal_flip : flips randomly horizontally 
#fill_mode : fills the hole when image is flipped

#convolutional model : algorithm to differentiate an image from another
#defining input shape
width = img_dims[0]
height = img_dims[1]
depth = img_dims[2]
inputShape = (height, width, depth)
dim = -1

#model
model = Sequential() #creates a sequential model of 5 hidden conv2d layers
model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))#32 hidden neurons 
model.add(Activation("relu")) #starts the neurons
model.add(BatchNormalization(axis=dim))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same")) #64 hidden neuron
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))

model.add(Conv2D(64, (3,3), padding="same")) #64 hidden neuron
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding="same")) #128 hidden neuron
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))

model.add(Conv2D(256, (3,3), padding="same")) #256 hidden neuron
model.add(Activation("relu"))
model.add(BatchNormalization(axis=dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten()) #transforms the data into a 1 dimensional array 

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("sigmoid"))

#compiles the model
opt = Adam(learning_rate = learning_rate)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

## fit the model
h = model.fit_generator(augmentation.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test,y_test),
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs, verbose=1)

## save the model
model.save('gender_predictor.model')


