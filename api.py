# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:54:22 2020

@author: chandrakumar
"""

from algorithm import *

from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    prediction = classifier.predict(test_image)
    return prediction.tolist()

if __name__ == '__main__':
    app.run()