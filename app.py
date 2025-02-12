import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

import traceback

print(f"TensorFlow Version: {tf.__version__}, Keras Version: {keras.__version__}")

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

        train_model = request.form.get('train_model')

        predict_pipeline = PredictPipeline(pred_range, train_model)

        dates, predictions = predict_pipeline.predict()

        print(predictions)
        print(dates)

        logging.info('Prediction values generated.')

        return render_template('home.html', predictions = predictions, dates = dates)

@app.errorhandler(500)
def internal_error(error):
    print("🔥 ERROR TRACEBACK 🔥")
    print(traceback.format_exc())  # Prints full error log in Render logs
    return "Internal Server Error", 500
  

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port = port, debug=True)