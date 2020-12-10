# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:01:56 2020

@author: chandrakumar
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from PIL import Image
from Preprocess import Pre_process as pp

class Model:
    
    @staticmethod
    def save_imgs():
        
        data_set = pp.loadImages(r'Data_Set/class1')
        res_set = pp.processing(data_set)       
        for i in range(len(res_set)):
            aimg = Image.fromarray(res_set[i])
            aimg=aimg.convert('RGB')
            aimg.save('Data/class1/'+str(i)+'.jpg','JPEG')
#            scipy.ndimage.imread(r'Data/class1/'+str(i)+'.jpg').convert('L')
        return 'preprocess is completed'
    
#
           
    @staticmethod
    def save_test():
       
        path = pp.loadImages(r'C:\Users\chandrakumar\Desktop\My_projcet')
        res_set = pp.processing(path)
        aimg = Image.fromarray(res_set[0])
        aimg = aimg.convert('RGB')
        aimg = aimg.save('test.jpg','JPEG')
        return aimg
       
        
       
    
    @staticmethod
    def model():
        
        classifier = Sequential()

        classifier.add(Convolution2D(32, 3, 3,input_shape = (256,256,3),activation='relu'))
        
        classifier.add(MaxPooling2D(pool_size=(2,2)))
        
        classifier.add(Flatten())
        
        classifier.add(Dense(units = 128, activation='relu'))
        
        classifier.add(Dense(units=1,activation='relu'))
        
        classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        
        from keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
        
        
        training_set = train_datagen.flow_from_directory(r'C:\Users\chandrakumar\Desktop\My_projcet\Data_Set',target_size=(256,256),batch_size=32,class_mode='binary')
        
        test_set = test_datagen.flow_from_directory(r'C:\Users\chandrakumar\Desktop\class_dataset\test_data', target_size=(256, 256), batch_size=32, class_mode='binary')
#        
        classifier.fit_generator(training_set,steps_per_epoch=20,epochs=2)
               
        Model.save_test()
        
        path = pp.loadImages(r'C:\Users\chandrakumar\Desktop\My_projcet')
        
        test_image = pp.processing(path)
               
        test_image = image.load_img(r'C:\Users\chandrakumar\Desktop\My_projcet\test.jpg', target_size=(256,256))
        #
        test_image = image.img_to_array(test_image)
        #
        #exp_img = image.img_to_array(training_set[0])
        #
        test_image = np.expand_dims(test_image, axis=0)
        #
        #exp_img = np.expand_dims(exp_img,axis=0)
        
        prediction = classifier.predict(test_image)
        
        
        
        print(prediction[0][0])

        return ""
        

