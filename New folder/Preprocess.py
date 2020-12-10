# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:43:05 2020

@author: chandrakumar
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy


class Pre_process:
    

    @staticmethod
    def loadImages(path):
                
        image_files = sorted([os.path.join(path,file) for file in os.listdir(path) if 
                              file.endswith('.jpg')])
        return image_files
    
    @staticmethod
    def display_one(a,title= 'original'):
        
        plt.imshow(numpy.real(a)),plt.title(title)
        plt.xticks([]),plt.yticks([])
        plt.show()
        
    @staticmethod   
    def display(a,b,title1='original',title2='edited'):
        
        plt.subplot(121),plt.imshow(numpy.real(a)),plt.title(title1)
        plt.xticks([]),plt.yticks([])
        plt.subplot(121),plt.imshow(numpy.real(b)),plt.title(title2)
        plt.xticks([]),plt.yticks([])
        plt.show()
    
    @staticmethod
    def processing(data):
        
        res_set = []
    
        img = [cv2.imread(i) for i in data]
        
        height =256
        width = 256
        dim = (width,height)
        res_img = []
        for i in range(len(img)):
            res = cv2.resize(img[i],dim,interpolation=cv2.INTER_LINEAR)
            res_img.append(res)
            
        no_noise = []
        for i in range(len(res_img)):
            blur = cv2.GaussianBlur(res_img[i],(5,5),0)
            no_noise.append(blur)
       
        imgs = no_noise
        for image in range(len(imgs)):
        
            gray= cv2.cvtColor(imgs[image],cv2.COLOR_RGB2GRAY)
            ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                # Further noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            
#            unknown = cv2.subtract(sure_bg, sure_fg)
            
            #Displaying segmented back ground
           
            res_set.append(sure_fg)
            
        return res_set
    


 
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    