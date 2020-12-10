# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:49:44 2020

@author: chandrakumar
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from alg2 import *

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3,input_shape = (256,256,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation='relu'))

classifier.add(Dense(output_dim=1,activation='relu'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Data',target_size=(256,256),batch_size=32,class_mode='binary')

#test_set = test_datagen.flow_from_directory('train_data', target_size=(256, 256), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set,steps_per_epoch=1,epochs=1)

import numpy as np
from keras.preprocessing import image
path = pp.loadImages(r'C:\Users\chandrakumar\Desktop\My_projcet')
test_image = preprocessing(path[0])

#test_image = image.load_img('test.jpg', target_size=(256,256))
#
test_image = image.img_to_array(test_image)
#
#exp_img = image.img_to_array(training_set[0])
#
test_image = np.expand_dims(test_image, axis=0)
#
#exp_img = np.expand_dims(exp_img,axis=0)

prediction = classifier.predict(test_image)

print(prediction)




