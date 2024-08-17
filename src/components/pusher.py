"""Module for pushing the model for predictions."""

import os
import mlflow
import tensorflow as tf
from src.logger import logging


class Pusher:
    """
    A class for pushing the trained model for predictions.

    This class provides methods to load a trained model and prepare it for
    predictions.

    Attributes:
        model_path (str): The path to the trained model directory.
        push_path (str): The path where the model will be available for
                         predictions.

    Methods:
        push(): Loads the trained model and makes it available for predictions.
    """

    def __init__(self, model_path: str, push_path: str):
        """
        Initializes a Pusher object with model and push paths.

        Parameters:
            model_path (str): The path to the trained model directory.
            push_path (str): The path where the model will be available for
                             predictions.
        """
        self.model_path: str = os.path.join(model_path, "model.keras")
        self.push_path: str = push_path

    def push(self):
        """
        Loads the trained model, logs its parameters and metrics using MLFlow,
        and saves it to the specified push path for future predictions.

        Parameters:
            None (uses instance variables model_path and push_path)

        Returns:
            None
        """
        os.makedirs(self.push_path, exist_ok=True)

        pusher_output = os.path.join(self.push_path, "model_pushed.keras")

        with mlflow.start_run(run_name="Model Push", nested=True):
            logging.info("Loading the model from %s", self.model_path)

            if os.path.exists(pusher_output):
                logging.info("Model already pushed to %s. Skipping push.",
                             self.push_path)
                mlflow.log_artifact(pusher_output)
                return

            try:
                model = tf.keras.models.load_model(self.model_path)
                logging.info("Model loaded successfully.")

                os.makedirs(self.push_path, exist_ok=True)
                model.save(pusher_output)
                logging.info("Model saved to %s", self.push_path)

                mlflow.log_artifact(pusher_output)
                logging.info("Model artifact logged to MLFlow.")

            except Exception as e:
                logging.error("Failed to load or push the model: %s", e)
                raise
