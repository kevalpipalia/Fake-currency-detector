# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:48:38 2020

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
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication(variance,skewness,curtosis,entropy):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
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
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction



if __name__=='__main__':
    app.run()