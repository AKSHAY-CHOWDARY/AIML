import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and scaler pickle
ridgeModel = pickle.load(open("..\\Project Algerian Forest Fires\\ridgeModel.pkl",'rb'))
scaler = pickle.load(open("..\\Project Algerian Forest Fires\\scaler.pkl","rb"))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
         
         Temperature = float(request.form['Temperature'])
         RH = float(request.form['RH'])
         Ws = float(request.form['Ws'])
         Rain = float(request.form['Rain'])
         FFMC = float(request.form['FFMC'])
         DMC = float(request.form['DMC'])
         ISI = float(request.form['ISI'])
         Classes = float(request.form['Classes'])
         Region = float(request.form['Region'])

         scaled_input = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
         prediction = ridgeModel.predict(scaled_input)

         return render_template('home.html',prediction = prediction[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")

