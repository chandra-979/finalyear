# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:38:05 2020

@author: chandrakumar
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy

image_path = r'C:\Users\chandrakumar\Desktop\My_projcet'


def loadImages(path):
    image_files = sorted([os.path.join(path,file) for file in os.listdir(path) if 
                          file.endswith('.jpg')])
    return image_files

def display_one(a,title= 'original'):
    plt.imshow(numpy.real(a)),plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.show()
    
def display(a,b,title1='original',title2='edited'):
    plt.subplot(121),plt.imshow(numpy.real(a)),plt.title(title1)
    plt.xticks([]),plt.yticks([])
    plt.subplot(121),plt.imshow(numpy.real(b)),plt.title(title2)
    plt.xticks([]),plt.yticks([])
    plt.show()

res_set = []
img_set = []

def processing(data):
    
#    display_one(data[0])
    
    img = [cv2.imread(i) for i in data]
    
#    print([i.shape for i in img ])
    
    height =256
    width = 256
    dim = (width,height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i],dim,interpolation=cv2.INTER_LINEAR)
        res_img.append(res)
#    print("resized image",i for i in res_img)
    
    original = res_img
#    display_one(original)
    for i in original:
        display_one(i)
    
    no_noise = []
    for i in range(len(res_img)):
        blur = cv2.GaussianBlur(res_img[i],(5,5),0)
        no_noise.append(blur)
    image = no_noise
    for i in range(len(image)):
        display(original[i],image[i],'original','blured')
    imgs = image
    
    
    for image in range(len(imgs)):
    
        gray= cv2.cvtColor(imgs[image],cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        display(original[image],thresh,"original","segmented")
        
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
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        #Displaying segmented back ground
        display(original[image], sure_bg, 'Original', 'Segmented Background')
        
        res_set.append(sure_bg)
        
#        # Marker labelling
#        ret, markers = cv2.connectedComponents(sure_fg)
#        
#        # Add one to all labels so that sure background is not 0, but 1
#        markers = markers + 1
#        
#        # Now, mark the region of unknown with zero
#        markers[unknown == 255] = 0
#        
#        markers = cv2.watershed(imgs[image], markers)
##        imgs[markers == -1] = [255, 0, 0]
#        
        # Displaying markers on the image
#        display(imgs[image], markers, 'Original', 'Marked')

def main():
    # calling global variable
    global image_path
    global img_set
    global res_set
    '''The var Dataset is a list with all images in the folder '''          
    dataset = loadImages(image_path)


     
    print("List of files the  in the folder:\n",dataset)
    print("--------------------------------")
    
    # sending all the images to pre-processing
    pro = processing(dataset)
    
   
    
#    from PIL import Image
#    
#    for i in range(len(res_set)):
#        aimg = Image.fromarray(res_set[i])
#        
#        aimg.save('Data/class1/'+str(i)+'.jpg','JPEG')
#    
# 
main()
    

    
    

    























    
    
    
    
    
    
    
    
    
    
    
    
    