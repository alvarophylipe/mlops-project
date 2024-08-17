""" Module containing the Trainer class. """

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import os
import mlflow
import pandas as pd
from src.logger import logging
from src.constants import (
    FIRST_DENSE_LAYER_UNITS,
    SECOND_DENSE_LAYER_UNITS,
    OUTPUT_DENSE_LAYER_UNITS,
    FIRST_DROPOUT_LAYER,
    SECOND_DROPOUT_LAYER,
    RELU_ACTIVATION,
    SIGMOID_ACTIVATION,
    EPOCHS, BATCH_SIZE,
    COMPILE_ARGS,
    EARLY_STOPPING_ARGS
)


class Trainer:
    """
    Class for training the model.
    """

    def __init__(self, input_path: str, model_path: str):
        """
        Initializes the Trainer class.

        Args:
            input_path (str): The path to the input CSV files.
            model_path (str): The path to the model directory.
        """
        self.input_path = input_path
        self.model_path = model_path
        self._train, self._test = self._read_train_test_csv()
        self.X_train, self.y_train, \
            self.X_test, self.y_test = self._split_samples()

    def train(self):
        """
        Trains the model.
        """
        os.makedirs(self.model_path, exist_ok=True)
        model_file_path = os.path.join(self.model_path, 'model.keras')

        try:
            with mlflow.start_run(run_name="Model Training", nested=True):
                if os.path.exists(model_file_path):
                    logging.info("Model already trained. Skipping training.")
                    mlflow.log_artifact(model_file_path)
                    return

                logging.info("Creating the model...")
                model = self._create_model()

                logging.info("Compiling the model...")
                model = self._compile_model(model)

                logging.info("Training the model...")
                early_stopping = EarlyStopping(**EARLY_STOPPING_ARGS)
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_test, self.y_test),
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[early_stopping]
                )

                logging.info("Training completed.")

                # Save the model
                model.save(model_file_path)
                mlflow.log_artifact(model_file_path)
                logging.info(f"Model saved to {model_file_path}")

                # Log training metrics
                self._log_training_metrics(history)

        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            raise

    def _create_model(self):
        """
        Creates the model.
        """
        model = Sequential([
            Input(shape=(self.X_train.shape[1],)),
            Dense(FIRST_DENSE_LAYER_UNITS, activation=RELU_ACTIVATION),
            Dropout(FIRST_DROPOUT_LAYER),
            Dense(SECOND_DENSE_LAYER_UNITS, activation=RELU_ACTIVATION),
            Dropout(SECOND_DROPOUT_LAYER),
            Dense(OUTPUT_DENSE_LAYER_UNITS, activation=SIGMOID_ACTIVATION)
        ])

        return model

    def _read_train_test_csv(self):
        """
        Reads the train and test CSV files.

        Returns:
            tuple: The train and test datasets as pandas DataFrames.
        """
        train = pd.read_csv(os.path.join(self.input_path, 'train.csv'))
        test = pd.read_csv(os.path.join(self.input_path, 'test.csv'))
        return train, test

    def _split_samples(self):
        """
        Splits the samples into features and target variables.

        Returns:
            tuple: The features and target variables for train and test sets.
        """
        X_train = self._train.drop('Attrition', axis=1)
        y_train = self._train['Attrition']
        X_test = self._test.drop('Attrition', axis=1)
        y_test = self._test['Attrition']
        return X_train, y_train, X_test, y_test

    def _compile_model(self, model):
        """
        Compiles the model.

        Args:
            model (Sequential): The Keras Sequential model to compile.

        Returns:
            Sequential: The compiled model.
        """
        model.compile(**COMPILE_ARGS)

        mlflow.log_params(COMPILE_ARGS)

        return model

    def _log_training_metrics(self, history):
        """
        Logs training metrics to MLflow.

        Args:
            history: The history object returned by the model's fit method.
        """
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history['loss'], history.history['accuracy'],
                history.history['val_loss'], history.history['val_accuracy'])):
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
