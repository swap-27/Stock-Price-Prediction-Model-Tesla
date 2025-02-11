from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

#Route for a homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        pred_range = request.form.get('prediction_duration')

        predict_pipeline = PredictPipeline(pred_range)

        dates, predictions = predict_pipeline.predict()

        print(predictions)
        print(dates)

        logging.info('Prediction values generated.')

        return render_template('home.html', predictions = predictions, dates = dates)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)