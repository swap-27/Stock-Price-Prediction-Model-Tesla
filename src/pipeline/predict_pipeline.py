import sys
import pandas as pd
from src.exception import CustomException
import keras
import datetime
from src.components import data_ingestion
import os
from src.utils import load_object
import numpy as np

class PredictPipeline:
    def __init__(self, pred_range:int):
        self.pred_range = int(pred_range)
    def predict(self):

        try:
            model_path = "artifacts/model.keras"
            preprocessor_path = "artifacts/preprocessor.pkl"

            chk_df = pd.read_csv('artifacts/data.csv')
            
            end_date = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
            end_date_main = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
            last_date = chk_df['Date'].iloc[[-1]].values[0]


            if chk_df['Date'].iloc[[-1]].values[0] != end_date_main:
                os.system("python src/components/data_ingestion.py")

            model = keras.models.load_model(model_path)

            scaler = load_object(preprocessor_path)

            test_data = pd.read_csv('artifacts/test.csv')
            test_data_scaled = scaler.transform(test_data['Close'].values.reshape(-1,1))

            time_steps = 60

            # Get the last `time_steps` days from the test set
            last_sequence = test_data_scaled[-time_steps:]

            # Reshape it for the model's input (it needs to be 3D: samples, time_steps, features)
            last_sequence = last_sequence.reshape(1, time_steps, 1)

            # Predict the next day (for the first prediction)
            next_day_prediction = model.predict(last_sequence)

            # Store predictions and recursively predict the next 7 days
            predictions = []
            pred_dates = []
            # Predict the next n days
            for i in range(self.pred_range):
                predictions.append(next_day_prediction[0, 0])
                
                last_date = (datetime.datetime.strptime(last_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

                pred_dates.append(last_date)

                # Update the input sequence with the newly predicted value
                last_sequence = np.append(last_sequence[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)
                
                # Predict the next day
                next_day_prediction = model.predict(last_sequence)

            # Invert the scaling to get the actual stock prices (reverse the MinMax scaling)
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

            predictions = predictions.flatten().tolist()

            predictions = [round(i, 2) for i in predictions]

            return pred_dates, predictions

        except Exception as e:
            raise CustomException(e, sys)
