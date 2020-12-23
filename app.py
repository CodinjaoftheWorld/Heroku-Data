import pandas as pd 
import numpy as np 
import pickle
from flask import Flask, request
import flasgger
from flasgger import Swagger
import joblib
import xgboost as xgb


app = Flask(__name__)
Swagger(app)    

# load model from file
regressor = pickle.load(open("regressor.pickle.dat", "rb"))


@app.route('/')
def welcome():
    return "Hello!!"

@app.route('/predict_file', methods=["POST"])
def predict_result():

    """Hey! Upload the test csv file here to get predictions  
    Click on "Try it out" to unable the file upload.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """

    # read the csv file 
    data_file = pd.read_csv(request.files.get("file"))
    # convert input into appropriate format acceptable by xgboost regressor model 
    pred_data = xgb.DMatrix(data = data_file)
    # do the predictions
    predictions = regressor.predict(pred_data)
    return "The predicted values for the csv is "+ str(list(predictions))


if __name__=="__main__":
    app.debug = True
    app.run()