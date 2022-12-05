#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Description: This program replicates the practice given at the following URL: https://github.com/josephlee94/intuitive-deep-learning
The program will implement a neural network in order to predict house prices based on a given CSV dataset.

Name: Andrea Marcelli
"""
import pandas as pd

# Importing the spreadsheet with the data into a data frame through pandas
df = pd.read_csv('housepricedata.csv')
# Display the dataframe
df


# In[2]:


# Converting the data frame into an array
dataset = df.values
# Display the array values
dataset


# In[3]:


# Splitting the dataset into input features and label to predict
X = dataset[:,0:10]
Y = dataset[:,10]

from sklearn import preprocessing

# Normalizing data to be inside the range between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_scale


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


# In[7]:


# Creating and Training the Neural Network
from keras.models import Sequential
from keras.layers import Dense

# Creating the first model with three layers, two hidden layers, and one output layer
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)), 
    Dense(32, activation='relu'), 
    Dense(1, activation='sigmoid'),
])

# Configuring the model by selecting algorithm to use, loss function, and metrics to track
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Storing the history of the data
hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))


# In[8]:


# Evalutating data
model.evaluate(X_test, Y_test)[1]


# In[9]:


import matplotlib.pyplot as plt

# Creating a plot to visualize the training loss and validation loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[16]:


# Creating a plot to visualize training accuracy and the validation accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[19]:


# Creating a model that will overfit
model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model_2.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
hist_2 = model_2.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))


# In[20]:


# Creating a plot to visualize the overfitting loss and validation loss
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[21]:


# Creating a plot to visualize the overfitting accuracy and validation accuracy
plt.plot(hist_2.history['accuracy'])
plt.plot(hist_2.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[30]:


# Creating a third model with L2 regularization and dropout incorporated
from keras.layers import Dropout
from keras import regularizers

model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
])


# In[31]:


model_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist_3 = model_3.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))


# In[32]:


# Creating a plot to visualize the trained loss and validation loss of the third model
plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()


# In[33]:


# Creating a plot to visualize the trained accuracy and validation accuracy of the third model
plt.plot(hist_3.history['accuracy'])
plt.plot(hist_3.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

