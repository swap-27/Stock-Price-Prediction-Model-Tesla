from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging
import tensorflow as tf
import keras
import sys
import traceback
import os

print(f"TensorFlow Version: {tf.__version__}, Keras Version: {keras.__version__}")

# Ensure only the correct paths are included
sys.path = [p for p in sys.path if "data science project" not in p]
print("Updated sys.path on Hugging Face:", sys.path)

# Confirm model file path
model_path = os.path.abspath("artifacts/model.h5")
print(f"Loading model from: {model_path}")

application = Flask(__name__)
app = application

LOCK_FILE = "training.lock"

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', predictions=None)
    
    pred_range = request.form.get('prediction_duration', 7)  # Ensure integer input
    train_model = request.form.get('train_model') is not None  # Convert checkbox to boolean

    if train_model:
        if os.path.exists(LOCK_FILE):
            return "Model training is already in progress. Please try again later.", 503

        try:
            with open(LOCK_FILE, 'x'):  # Ensure lock file is unique
                print("Training started...")
                predict_pipeline = PredictPipeline(pred_range, train_model)
                predictions = predict_pipeline.predict()
        finally:
            os.remove(LOCK_FILE)  # Remove lock after training
    else:
        predict_pipeline = PredictPipeline(pred_range, train_model)
        predictions = predict_pipeline.predict()

    print(predictions)
    logging.info('Prediction values generated.')

    return render_template('home.html', predictions=predictions)

@app.errorhandler(500)
def internal_error(error):
    print("ERROR TRACEBACK")
    print(traceback.format_exc())
    return "Internal Server Error", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  
    app.run(host="0.0.0.0", port=port, debug=True)
