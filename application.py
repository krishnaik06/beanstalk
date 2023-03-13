import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app_logger import log
import warnings
warnings.filterwarnings("ignore")

application = Flask(__name__) # initializing a flask app
app=application

# Import Classification and Regression model file
transformer=bz2.BZ2File('model/transformer.pkl', 'rb')
model_T=pickle.load(transformer)
R_pickle_in = bz2.BZ2File('model/regression.pkl', 'rb')
model_R = pickle.load(R_pickle_in)

# Route for homepage
@app.route('/')
def index():
    log.info('Home page loaded successfully')
    return render_template('index1.html')

# Route for Single Prediction
@app.route('/single_regression',methods=['GET','POST'])
def single_regression():
    log.info('Single Prediction loaded successfully')
    if request.method =='POST':
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Region = float(request.form.get('classes'))
            result = model_R.predict([[RH,Ws,Rain,FFMC,DMC,ISI,Region]])
            log.info('Single Prediction done successfully', result)
            return render_template('single_prediction.html', result= result)
    
    return render_template('single_prediction.html')

# Route for API Testing
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        log.info('Input from Api testing', data)
        new_data = [list(data.values())]
        final_data = model_T.transform(new_data)
        output = model_R.predict(final_data)[0]
        return jsonify(f"Temprature will be: {output}")
    except Exception as e:
        output = 'Check the in input again!'
        log.error('error in input from Postman', e)
        return jsonify(output)


# Run APP in Debug mode
if __name__ == "__main__":
    app.run(host="0.0.0.0")
