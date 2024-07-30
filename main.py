# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
#import pickle
#import random
import os

# Defining parameters
path = "data"
classFile = "labels.csv"
batch_size_val = 32
epochs_val = 10
imageDim = (32,32,3)  # 3 channels - RGB
testRatio = 0.2
valRatio = 0.2

ClassIdCount = 0
images = []
classId = []
myList = os.listdir(path)       # list all items in the Data dictionary
print("Total classes detected: ", len(myList))      # Stores items of different classes i.e. 0, 1, 2..., 57
numClasses = len(myList)
for x in range(0, numClasses):
    myPicList = os.listdir(path+"/"+str(ClassIdCount))     # Stores the list of file names in each directory inside Data. For ex, myPicList starts to store all image filenames from directory 0 and so on...
    for y in myPicList:
        full_path = os.path.join(path, str(ClassIdCount), y)       # Constructs the full path to teh correct image, i.e. Data/0/000_1_0001.png
        curImg = cv2.imread(full_path)      # reads the image from the constructed path full_path and stores it in curImage
        if curImg is not None:
            curImg = cv2.resize(curImg, (imageDim[0], imageDim[1]))
            images.append(curImg)
            classId.append(ClassIdCount)
        else:
            print("Error loading the image: ", full_path)
            
    ClassIdCount += 1
print(" ")

images = np.array(images)
classId = np.array(classId)

print("Images shape:", images.shape)
print("Class IDs shape:", classId.shape)

# Train, Validation and test split of data
X_train, X_test, y_train, y_test = train_test_split(images, classId, test_size = testRatio, random_state = 52)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = valRatio, random_state = 52)

print(f"Train: X: {X_train.shape}, y: {y_train.shape}")
print(f"Validation: X: {X_val.shape}, y: {y_val.shape}")
print(f"Test: X: {X_test.shape}, y: {y_test.shape}")

# Reading the labels
label = pd.read_csv("labels.csv")
print("Shape of the data: ", label.shape)

# Pre-processing of data, i.e. images
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)  # Equalize the histogram
    img = img.astype('float32')  # Convert to float32
    img /= 255      # normalizing the img to range from 0 to 1
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)  # no. of samples in the train dataset, height, width of the images, single channel (if RGB then - 3)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Generating more images by manipulating existing ones in X_train
dataAug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataAug.fit(X_train)
batches = dataAug.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

y_train = to_categorical(y_train, numClasses)
y_val = to_categorical(y_val, numClasses)
y_test = to_categorical(y_test, numClasses)

def cnnModel():
    model = Sequential()
    # inout layer - 60 kernel, 5x5 filter
    model.add((Conv2D(60,(5,5), input_shape=(imageDim[0], imageDim[1], 1), activation='relu')))
    model.add((Conv2D(60,(5,5), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add((Conv2D(30,(3,3),activation='relu')))
    model.add((Conv2D(30,(3,3), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))     # To avoid overfitting of the model
 
    model.add(Flatten())        # Convert to single array
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(numClasses,activation='softmax'))   # multiclass - softmax (if binary classification then sigmoid)
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])      # multiclass - categorical_crossentropy (if binary then binary_crossentropy)
    return model

model = cnnModel()
print(model.summary())
# Can use fit_generator (deprecated) as we are loading the data from ImageDataGenerator
history=model.fit(dataAug.flow(X_train,y_train,batch_size=32),steps_per_epoch=len(X_train)//32,epochs=epochs_val,validation_data=(X_val,y_val),shuffle=1)

# Accuracy curve
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score: ', score[0])
print('Test Accuracy: ', score[1])

model.save("model.h5")      # Saving the model