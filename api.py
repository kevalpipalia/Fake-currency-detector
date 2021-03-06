# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:27:47 2020

@author: keval
"""


import pandas as pd
import numpy as np
from flask import Flask,request
import pickle


app = Flask(__name__)
pkl_in = open('classifier.pkl','rb')
classifier = pickle.load(pkl_in)

@app.route('/')
def welcome():
    return "Welcome to UWash"

@app.route('/predict')
def predict_note():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    pred = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "Your note measurements predicts it is" + str(pred)

@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    pred = classifier.predict(df_test) 
    return "Your note measurements predicts it is" + str(list(pred))

if __name__ == '__main__':
    app.run()
