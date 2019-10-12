import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

size, depth =100 , 30

result = []

for i in range(6):
   if (i==0):
       path = './drive/My Drive/Project/Dataset/boxing'
       print('Extrecting frames in boxing class: '+path)
   elif (i==1):
       path = './drive/My Drive/Project/Dataset/handclapping'
       print('Extrecting frames in handclapping class: '+path)
   elif (i==2):
       path = './drive/My Drive/Project/Dataset/handwaving'
       print('Extrecting frames in handwaving class: '+path)
   elif (i==3):
       path = './drive/My Drive/Project/Dataset/jogging'
       print('Extrecting frames in jogging class: '+path)
   elif (i==4):
       path = './drive/My Drive/Project/Dataset/running'
       print('Extrecting frames in jogging class: '+path)
   else:
       path = './drive/My Drive/Project/Dataset/walking'
       print('Extrecting frames in walking class: '+path)    
   listing = os.listdir(path)
   for vid in listing:
        vid = path+'/'+vid
        frames = []
        cap = cv2.VideoCapture(vid)
        fps = cap.get(5)
        for k in range(depth):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(size,size),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
        inp = np.array(frames)        
        ipt=np.rollaxis(np.rollaxis(inp,2,0),2,0)
        print(ipt.shape)
        result.append(ipt)
result_array = np.array(result)
num_samples = len(result_array)
print (num_samples)
# Assign Label to each class
label=np.ones((num_samples,), dtype = int)
label[0:100] = 0
label[100:199] = 1
label[199:299] = 2
label[299:399] = 3
label[399:499]= 4
label[499:] = 5

train_data = [result_array, label]
# X_train is now X_tr_array
# y_train is now label
(X_train, y_train) = (train_data[0],train_data[1])
# 599 num_samples, 16,16,15 shape
print('X_Train shape:', X_train.shape)
print('y_train shape: ', y_train.shape)

train_set = np.zeros((num_samples, size, size, depth, 1))
X_train2=np.reshape(X_train,(num_samples, size, size, depth, 1))
train_set=X_train2
print(train_set.shape, 'train samples')

nb_classes = 6
from keras.utils import np_utils, generic_utils
Y_train = np_utils.to_categorical(y_train, nb_classes)

# Pre-processing
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /= np.max(train_set)

# Define model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv3D(64, kernel_size= (3, 3, 3), padding='same', input_shape=(size, size, depth, 1), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling3D(pool_size=(3, 3, 3)))
# Step 1 - Convolution
classifier.add(Conv3D(32, kernel_size= (3, 3, 3), padding='same', activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling3D(pool_size=(3, 3, 3)))

# Step 1 - Convolution
#classifier.add(Conv3D(32, kernel_size= (3, 3, 3), padding='same', activation='relu'))

# Step 2 - Pooling
#classifier.add(MaxPooling3D(pool_size=(3, 3, 3)))


#step 7 - Flatten
classifier.add(Flatten())

#step 8 - Full collection first leyare 
classifier.add(Dense(units = 30, kernel_initializer='normal', activation='relu'))

#step 9 - Dropout
classifier.add(Dropout(0.5))

#classifier.add(Dense(units = 30, kernel_initializer='normal', activation='relu'))

#classifier.add(Dropout(0.3))
#step 10 - output layer
classifier.add(Dense(units = 6, kernel_initializer='normal', activation = 'softmax'))

#step 11 - complie the model
classifier.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics = ['accuracy'])

#sumery

classifier.summary()

# Split the data into test and trainng set
from sklearn.model_selection import train_test_split
X_train_new, X_test_new, y_train_new, y_test_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=0)

hist = classifier.fit(X_train_new,
                      y_train_new,
                      validation_data=(X_test_new,y_test_new),
                      batch_size=32,
                      epochs = 150,
                      shuffle=True)   

# Train the model
classifier.save('./drive/My Drive/Project/92_acc.h5')

 # Evaluate the model
score = classifier.evaluate(X_test_new, y_test_new, batch_size=2 )
print('Test score:', score[0])
print('Test accuracy:', score[1]) 

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(150)

plt.figure(1,figsize=(6,3))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['ggplot'])

plt.figure(2,figsize=(6,3))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['ggplot'])

from sklearn.metrics import confusion_matrix
Y_prediction = classifier.predict(X_test_new)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_prediction,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test_new,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(confusion_mtx, annot=True)