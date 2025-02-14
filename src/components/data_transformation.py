from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import sys
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:

            scaler = MinMaxScaler(feature_range=(0,1))

            return scaler
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_sequences(self, data, time_steps=60):
        try:
            x, y = [], []
            for i in range(time_steps, len(data)):
                x.append(data[i - time_steps:i,0])
                y.append(data[i, 0])
            return np.array(x), np.array(y) 
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info('Read Train and Test data successfully')
            
            train_data = train_data.set_index('Date', inplace=True)
            test_data = test_data.set_index('Date', inplace=True)

            scaler = self.get_data_transformer_object()
            train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1,1))

            test_data_scaled = scaler.transform(test_data.values.reshape(-1,1))


            time_steps = 60  # Use the past 60 days to predict the next day's price
            x_train, y_train = self.create_sequences(train_data_scaled, time_steps)
            x_test, y_test = self.create_sequences(test_data_scaled, time_steps)

            # Reshape the input to 3D format: (samples, time_steps, features)
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)  # (samples, time_steps, features)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)  # (samples, time_steps, features)

            logging.info("Data transformation completed successfully.")

            # Save MinMaxScaler for future use
            save_object(self.data_transformation_config.preprocessor_obj_file_path, scaler)

            return x_train, x_test, y_train, y_test, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)