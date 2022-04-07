#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import zipfile 
import tensorflow as tf 
import tensorflow.keras as k
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import datasets,layers 
from tensorflow.keras import Model 
from tensorflow.keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import time as time
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input,Dense, Dropout, Conv2D, MaxPool2D, Flatten
import tensorflow_hub as hub
import PIL.Image as Image
from sklearn.utils import shuffle


# In[34]:


from tensorflow.keras.layers import Lambda,Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# In[35]:


# [DO NOT MODIFY THIS CELL]

# load the images
n_img = 50000
n_noisy = 40000
n_clean_noisy = n_img - n_noisy
imgs = np.empty((n_img,32,32,3))
for i in range(n_img):
    img_fn = f'images/{i+1:05d}.png'
    imgs[i,:,:,:]=cv2.cvtColor(cv2.imread(img_fn),cv2.COLOR_BGR2RGB)

# load the labels
clean_labels = np.genfromtxt('clean_labels.csv', delimiter=',', dtype="int8")
noisy_labels = np.genfromtxt('noisy_labels.csv', delimiter=',', dtype="int8")


# In[36]:


# [DO NOT MODIFY THIS CELL]

fig = plt.figure()

ax1 = fig.add_subplot(2,4,1)
ax1.imshow(imgs[0]/255)
ax2 = fig.add_subplot(2,4,2)
ax2.imshow(imgs[1]/255)
ax3 = fig.add_subplot(2,4,3)
ax3.imshow(imgs[2]/255)
ax4 = fig.add_subplot(2,4,4)
ax4.imshow(imgs[3]/255)
ax1 = fig.add_subplot(2,4,5)
ax1.imshow(imgs[4]/255)
ax2 = fig.add_subplot(2,4,6)
ax2.imshow(imgs[5]/255)
ax3 = fig.add_subplot(2,4,7)
ax3.imshow(imgs[6]/255)
ax4 = fig.add_subplot(2,4,8)
ax4.imshow(imgs[7]/255)

# The class-label correspondence
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# print clean labels
print('Clean labels:')
print(' '.join('%5s' % classes[clean_labels[j]] for j in range(8)))
# print noisy labels
print('Noisy labels:')
print(' '.join('%5s' % classes[noisy_labels[j]] for j in range(8)))


# In[37]:


# [DO NOT MODIFY THIS CELL]
# RGB histogram dataset construction
no_bins = 6
bins = np.linspace(0,255,no_bins) # the range of the rgb histogram
target_vec = np.empty(n_img)
feature_mtx = np.empty((n_img,3*(len(bins)-1)))
i = 0
for i in range(n_img):
    # The target vector consists of noisy labels
    target_vec[i] = noisy_labels[i]
    
    # Use the numbers of pixels in each bin for all three channels as the features
    feature1 = np.histogram(imgs[i][:,:,0],bins=bins)[0] 
    feature2 = np.histogram(imgs[i][:,:,1],bins=bins)[0]
    feature3 = np.histogram(imgs[i][:,:,2],bins=bins)[0]
    
    # Concatenate three features
    feature_mtx[i,] = np.concatenate((feature1, feature2, feature3), axis=None)
    i += 1


# In[38]:


# [DO NOT MODIFY THIS CELL]
# Train a logistic regression model 
clf = LogisticRegression(random_state=0).fit(feature_mtx, target_vec)


# In[39]:


# [DO NOT MODIFY THIS CELL]
def baseline_model(image):
    '''
    This is the baseline predictive model that takes in the image and returns a label prediction
    '''
    feature1 = np.histogram(image[:,:,0],bins=bins)[0]
    feature2 = np.histogram(image[:,:,1],bins=bins)[0]
    feature3 = np.histogram(image[:,:,2],bins=bins)[0]
    feature = np.concatenate((feature1, feature2, feature3), axis=None).reshape(1,-1)
    return clf.predict(feature)


# In[40]:


X_train, X_test,y_train,y_test = train_test_split(imgs,noisy_labels,test_size = 0.2,random_state = 1234)


# In[41]:


#X_train = X_train.reshape(40000, 1024,3)
#X_test = X_test.reshape(10000, 1024,3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[42]:


#one-hot-encoding
n_classes = 10
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)


# In[74]:


# building a linear stack of layers with the sequential model
def model1():
    model = tf.keras.Sequential([
        
        # convolutional layer
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPool2D(pool_size=(1,1)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPool2D(pool_size=(1,1)),
    tf.keras.layers.Dropout(0.5),
    # flatten output of conv
    tf.keras.layers.Flatten(),
    # hidden layer
    tf.keras.layers.Dense(100, activation='relu'),
    # output layer
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

    # looking at the model summary
    #model.summary()
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    # training the model for 10 epochs
    #model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs = 10 ):
    
    # generate image data with data augmentation 
    train_gen = ImageDataGenerator(
        featurewise_center=True, # set the mean of the inputs to 0 over the dataset
        featurewise_std_normalization=True, # divide the inputs by standard deviation of the dataset
        rotation_range=20, # degree range for random rotations
        width_shift_range=0.1, # the fraction of total width
        height_shift_range=0.1, # the fraction of total height
        horizontal_flip=True) # flip the inputs horizontally randomly
    
    train_gen.fit(X_train)
    
    test_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    test_gen.fit(X_train)
    
    
    # fits the model on batches with data augmentation:
    model.fit(train_gen.flow(X_train, y_train, batch_size=128),
             validation_data=train_gen.flow(X_test, y_test, batch_size=12),
            
              epochs=epochs)
    
    return model, test_gen 

def model_I(image):
    '''
    This function should takes in the image of dimension 32*32*3 as input and returns a label prediction
    '''
    
    # predict
    pred = model1.predict(data_genI.flow(image))
    pred_class = np.argmax(pred)
    
    return pred_class


# In[75]:


model = model1()
model, data_genI = train_model(model, X_train, y_train, X_test, y_test, epochs = 10)


# In[76]:


train_metrics = model.evaluate(data_genI.flow(X_train, y_train))
test_metrics = model.evaluate(data_genI.flow(X_test, y_test))

print(f"Model I Training Loss, Accuracy: {train_metrics}")
print(f"Model I Testing Loss, Accuracy: {test_metrics}")


# In[77]:


# clean images and labels
img_cl = imgs[:10000]
y_cl = np.eye(10)[clean_labels]

# estimate the model accuracy using clean labels
cl_metrics = model.evaluate(data_genI.flow(img_cl, y_cl))

print(f"Model I Clean Labels Loss, Accuracy: {cl_metrics}")


# In[78]:


clean = np.eye(10)[clean_labels]
noisy = np.eye(10)[noisy_labels[:10000]]
clean_imgs = imgs[:10000]/255


# In[79]:


def clean_l():
    img_input = Input(shape=(32, 32, 3))
    noisy_label = Input(shape = (10))
    
    vgg = VGG16(input_shape = (32,32,3),weights='imagenet',include_top=False,pooling='max')
    vgg.trainable = False
    vgg.get_layer('block5_conv3').trainable = True
    img_vec = vgg(img_input)
    
    noisy_l = Dense(10)(noisy_label)
    img_vec = Dense(512)(img_vec)
    
    x = Concatenate(axis=-1)([noisy_l, img_vec])
    x = Flatten(name='flatten')(x)
    x = Dense(50, activation = 'relu',name='fc1')(x)
    x = Dense(20, activation = 'relu',name='fc2')(x)
    out = Dense(10, activation = 'softmax',name='prediction')(x)
    
    model = Model([img_input, noisy_label], out)
    model.compile(loss='categorical_crossentropy',metrics=['acc'], optimizer='adam')
    
    return model


# In[80]:


model = clean_l()


# In[82]:


model.fit([clean_imgs, noisy], clean, batch_size = 256, epochs = 10)


# In[83]:


noisy_imgs = imgs[10000:]/255
noisy_l = np.eye(10)[noisy_labels[10000:]]


# In[84]:


new_pred = model.predict([noisy_imgs,noisy_l])


# In[85]:


#clean up label vectors
row_maxes = new_pred.argmax(axis=1)
new_labels = np.eye(10)[row_maxes]

#Create new train set from clean images and new pred labels
upd_imgs = imgs
upd_labels = np.concatenate((np.eye(10)[clean_labels], new_labels), axis=0) 


# In[86]:


X_train, X_test, y_train, y_test = train_test_split(upd_imgs,upd_labels, test_size=0.20, random_state=1234)


# In[87]:


model2 = model1()
modelII, data_genII = train_model(model2, X_train, y_train, X_test, y_test, 10)


# In[88]:


# clean images and labels
img_cl = imgs[:10000]
y_cl = np.eye(10)[clean_labels]

# estimate the model accuracy using clean labels
cl_metrics = modelII.evaluate(data_genII.flow(img_cl, y_cl))
print(f"Model II Clean Labels Loss, Accuracy: {cl_metrics}")


# In[73]:


# [DO NOT MODIFY THIS CELL]
def evaluation(model, test_labels, test_imgs):
    y_true = test_labels
    y_pred = []
    for image in test_imgs:
        y_pred.append(model(image))
    print(classification_report(y_true, y_pred))


# In[ ]:


# [DO NOT MODIFY THIS CELL]
# This is the code for evaluating the prediction performance on a testset
# You will get an error if running this cell, as you do not have the testset
# Nonetheless, you can create your own validation set to run the evlauation
n_test = 10000
test_labels = np.genfromtxt('clean_labels.csv', delimiter=',', dtype="int8")

test_imgs = np.empty((n_test,32,32,3))
for i in range(n_test):
    img_fn = f'images/{i+1:05d}.png'
    test_imgs[i,:,:,:]=cv2.cvtColor(cv2.imread(img_fn),cv2.COLOR_BGR2RGB)

evaluation(baseline_model, test_labels, test_imgs)

