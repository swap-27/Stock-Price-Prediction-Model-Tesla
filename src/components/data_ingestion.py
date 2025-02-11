import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os
import sys
from src.components.fetch_stock_data import StockPriceDataFetch
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import datetime
from dataclasses import dataclass 
import traceback

import pandas as pd
import math
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Method")

        try:
            
            stock_price_data_fetch = StockPriceDataFetch()
            close_df = stock_price_data_fetch.close_df

            logging.info("Train Test split initiated")
            train_data, test_data = train_test_split(close_df, test_size=0.2, shuffle=False)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            close_df.to_csv(self.ingestion_config.raw_data_path, index = True)

            train_data.to_csv(self.ingestion_config.train_data_path, index = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            print("Full error traceback:")
            print(traceback.format_exc())  # Prints full error details
    
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    
    data_transformation = DataTransformation()

    X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()

    model_trainer.initiate_model_trainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    
    