# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:24:49 2020

@author: chandrakumar
"""

from flask import Flask
import requests
import shutil
from Model import Model as M


app = Flask(__name__)


@app.route('/')
def home():
   o = M.save_imgs()
   return 'Automatic Aseesement Of Pavement Condition Bases On Photograph'+"  "+o
   
    

@app.route('/upload')
def take_Input():
    
    image_url = "https://tse1.mm.bing.net/th?id=OIP.MY7EAROiakb2Vh2Nsrh2fgHaCu&pid=Api&rs=1&c=1&qlt=95&w=263&h=96"
    
    # Open the url image, set stream to True, this will return the stream content.
    resp = requests.get(image_url, stream=True)
    # Open a local file with wb ( write binary ) permission.
    local_file = open('test.jpg', 'wb')
    # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
    resp.raw.decode_content = True
    # Copy the response stream raw data to local image file.
    shutil.copyfileobj(resp.raw, local_file)
    # Remove the image url response object.
    del resp
    return 'input is taken'

@app.route('/upload/output')
def final_output():
    op = M.model()
    return 'here the final o/p'

    
    

if __name__ == '__main__':
    app.run()