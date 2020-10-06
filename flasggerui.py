# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 21:35:01 2020

@author: keval
"""

import pandas as pd
import numpy as np
from flask import Flask,request
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pkl_in = open('classifier.pkl','rb')
classifier = pickle.load(pkl_in)

@app.route('/')
def welcome():
    return "Welcome to UWash"

@app.route('/predict')
def predict_note():
    
    """Fake Currency note detection System
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responces:
        200:
            description: The prediction is
        
    """
    
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    pred = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "Your note measurements predicts it is" + str(pred)

@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Fake currency note detection System
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
             The predictions are
    """
        
    df_test = pd.read_csv(request.files.get("file"))
    pred = classifier.predict(df_test) 
    return "Your note measurements predicts it is" + str(list(pred))

if __name__ == '__main__':
    app.run()
