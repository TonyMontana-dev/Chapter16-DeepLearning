#!/usr/bin/env python
# coding: utf-8

# In[93]:


"""
Description: This program replicates the practice given at the following URL: https://github.com/josephlee94/intuitive-deep-learning
The program will implement a neural network in order to recognize images of the CIFAR-10 dataset.

Name: Andrea Marcelli
"""
# Importing SSL to solve the issue related to Security Certificate of the dataset CIFAR10
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[94]:


print('x_train shape:', x_train.shape)


# In[95]:


print('y_train shape:', y_train.shape)


# In[96]:


# Display pixels of an image
print(x_train[0])


# In[97]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


# Displaying an image
img = plt.imshow(x_train[0])


# In[99]:


# Displaying the label of the image
print('The label is:', y_train[0])


# In[100]:


img = plt.imshow(x_train[1])


# In[101]:


print('The label is:', y_train[1])


# In[102]:


import keras
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)


# In[103]:


print('The one hot label is:', y_train_one_hot[1])


# In[104]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255


# In[105]:


x_train[0]


# In[106]:


# Creating a model building and training the neural network (Using Conventional 2D tensor and MaxPooling layers)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# In[107]:


model = Sequential()


# In[108]:


model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))


# In[109]:


model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))


# In[110]:


model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()


# In[111]:


# Configuring the algorithm, the loss function, and the metrics to track
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Training our neural network with a batch size of 32 and 20 epochs, splitting the dataset using validation split(20%).
hist = model.fit(x_train, y_train_one_hot, batch_size=32, epochs=20, validation_split=0.2)


# In[112]:


# Creating a plot to visualize the losses and validation loss of the model
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[113]:


# Creating a plot to visualize the accuracy and validation accuracy of the model
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[114]:


model.evaluate(x_test, y_test_one_hot)[1]


# In[115]:


# Saving the model
model.save('my_cifar10_model.h5')


# In[116]:


# Importing local image
my_image = plt.imread("cat.jpg")


# In[117]:


# Resizing the image
from skimage.transform import resize
my_image_resized = resize(my_image, (32,32,3))


# In[118]:


img = plt.imshow(my_image_resized)


# In[119]:


# Store into a variable the predictionof the resized image
import numpy as np
probabilities = model.predict(np.array([my_image_resized,]))


# In[120]:


# Display the probabilities rates
probabilities


# In[122]:


# Predicting the image identity
number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])

