import sys
import pandas as pd
from src.exception import CustomException
import datetime
from src.components import data_ingestion
import os
from src.utils import load_object
import numpy as np
import tensorflow as tf
from keras.models import load_model
from src.components.data_transformation import DataTransformation
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

class PredictPipeline:
    def __init__(self, pred_range:int, train_model):
        self.pred_range = int(pred_range)
        self.train_model = train_model

        #delete from here
        model_path = "artifacts/model.h5"
        preprocessor_path = "artifacts/preprocessor.pkl"

        with tf.device('/CPU:0'):
            self.model = load_model(model_path)

        self.scaler = load_object(preprocessor_path)
        self.test_data = pd.read_csv('artifacts/test.csv')



    def predict(self):

        try:
            model_path = "artifacts/model.h5"
            preprocessor_path = "artifacts/preprocessor.pkl"
            graph_output_path="stock_predictions.jpg"

            chk_df = pd.read_csv('artifacts/data.csv')
            
            end_date_main = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
            
            
            last_date = chk_df.iloc[-1]['Date']


            if self.train_model:
                os.system("python src/components/data_ingestion.py")

            
            with tf.device('/CPU:0'):
                model = load_model(model_path)
                print("Model loaded successfully!")

            scaler = load_object(preprocessor_path)

            test_data = pd.read_csv('artifacts/test.csv')

            test_data['Date'] = test_data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            test_data.set_index('Date', inplace=True)
            test_data_scaled = scaler.transform(test_data['Close'].values.reshape(-1,1))

            time_steps = 60

            datatrans = DataTransformation()
            x_test, y_test = datatrans.create_sequences(test_data_scaled, time_steps)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
            hist_pred = model.predict(x_test)
            hist_pred = scaler.inverse_transform(hist_pred)
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            plt.figure(figsize=(10, 6))

            # Plot actual vs predicted prices for the test set
            plt.plot(test_data.index[time_steps:], y_test_actual, color='blue', label='Actual Stock Price')
            plt.plot(test_data.index[time_steps:], hist_pred, color='red', label='Predicted Stock Price')

            # Labeling the graph
            plt.title('Stock Price Prediction Comparison - Predicted vs. Actual')
            
            plt.ylabel('Tesla - Stock Price')
            plt.legend()

            # Display the plot with proper date formatting
            plt.xticks(rotation=30)
            static_dir = os.path.join(os.getcwd(), 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
            image_path = os.path.join(static_dir, 'stock_predictions.jpg')
            plt.savefig(image_path, format='jpeg', dpi=300)
            plt.close()
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

            pred_df = pd.DataFrame(zip(pred_dates, predictions), columns=['Date','Predicted Price'])
            print(pred_df)

            pred_df_img = pred_df.copy()

            pred_df_img['Date'] = pred_df_img['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
            pred_df_img.set_index('Date', inplace=True)
            plt.figure(figsize=(10, 6))
            plt.plot(pred_df_img, color='red', label='Predicted Stock Price')

            # Labeling the graph
            plt.title('Tesla - Predicted Stock Price ($) for Next '+str(self.pred_range)+' Days')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.xticks(rotation=30)
            plt.legend()
            pred_image_path = os.path.join(static_dir, 'future_stock_predictions.jpg')
            plt.savefig(pred_image_path, format='jpeg', dpi=300)
            plt.close()
            
            return pred_df

        except Exception as e:
            raise CustomException(e, sys)