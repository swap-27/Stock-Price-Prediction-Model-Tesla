import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dataclasses import dataclass
import os
from src.logger import logging
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
import sys
from keras.callbacks import EarlyStopping
from src.exception import CustomException
from src.utils import save_object
import joblib


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Split training and test input")
            model = Sequential()

            # Add LSTM layers
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # LSTM with 50 units
            model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

            model.add(LSTM(units=50, return_sequences=False))  # Second LSTM layer
            model.add(Dropout(0.2))  # Dropout layer

            # Add a Dense layer for output
            model.add(Dense(units=1))  # Output layer with 1 unit (next day's stock price)

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])


            model.save("artifacts/model.keras", save_format="keras")
            # Evaluate the model on the test set
            test_loss = model.evaluate(X_test, y_test)
            print(f"Test Loss: {test_loss}")

        except Exception as e:
            raise CustomException(e, sys)