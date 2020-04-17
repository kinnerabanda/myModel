#!/usr/bin/env python
# coding: utf-8

# In[14]:


import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, AveragePooling2D
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


# In[15]:


#(input_train, output_train), (input_test, output_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

#print("Image Shape: {}".format(x_train[0].shape), end = '\n\n')


# In[17]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(30, kernel_size=(5, 5), padding='valid', activation='tanh', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(tf.keras.layers.Conv2D(13, kernel_size=(3,3), padding='valid', activation='tanh'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='tanh'))
model.add(tf.keras.layers.Dense(86, activation='tanh'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[18]:


#model.compile(optimizer=optimizers.sgd(lr=0.01, momentum=0.9),
              #loss='categorical_crossentropy',
              #metrics=['accuracy'])
#loss='categorical_crossentropy'
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


#history = model.fit(input_train/255, to_categorical(output_train), epochs=20, batch_size=32)
hist = model.fit(x_train,
         to_categorical(y_train),
         batch_size=32,
         epochs=62, #make it 62.
         validation_data=(x_test, to_categorical(y_test)))


# In[35]:


#score = model.evaluate(input_test/255, to_categorical(output_test), batch_size=32)
#score = model.evaluate(x_test, y_test, verbose=0)


# In[10]:


#print("Testset Loss: %f" % score[0])
#print("Testset Accuracy: %f" % score[1])


# In[11]:


#predictions = model.predict(input_test)
#y_pred = np.argmax(predictions, axis=1)
#print('Test set error rate: {}'.format(np.mean(y_pred == output_test)))


# In[78]:


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
plt.savefig('foo.png')


# In[79]:


_, acc = model.evaluate(x_test, to_categorical(y_test), verbose=0)
print('> %.3f' % (acc * 100.0))


# In[ ]:




