""" Module containing the Evaluation class. """

from tensorflow.keras.models import load_model  # type: ignore
import os
import mlflow
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
from src.logger import logging


class Evaluator:
    """
    Class for evaluating the model.
    """

    def __init__(self, model_path: str, test_data_path: str, output_path: str):
        """
        Initializes the Evaluation class.

        Args:
            model_path (str): The path to the saved model directory.
            test_data_path (str): The path to the test CSV file.
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.output_path = output_path
        self.model = self._load_model()
        self.X_test, self.y_test = self._load_test_data()

    def evaluate(self):
        """
        Evaluates the model on the test data and logs metrics to MLflow.
        """
        os.makedirs(self.output_path, exist_ok=True)

        output_path = os.path.join(self.output_path, "metrics.json")

        try:
            with mlflow.start_run(run_name="Model Evaluation", nested=True):
                logging.info("Evaluating the model...")

                if os.path.exists(output_path):
                    logging.info("Model evaluation already completed. \
                        Skipping evaluation.")
                    mlflow.log_artifact(output_path)
                    return

                # Make predictions
                predictions = self.model.predict(self.X_test).ravel()
                predictions = (predictions > 0.5).astype(int)

                # Calculate metrics
                accuracy = accuracy_score(self.y_test, predictions)
                precision = precision_score(self.y_test, predictions)
                recall = recall_score(self.y_test, predictions)
                f1 = f1_score(self.y_test, predictions)
                roc_auc = roc_auc_score(self.y_test, predictions)

                # Log metrics
                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc_score": roc_auc
                }

                joblib.dump(metrics, output_path)

                mlflow.log_metrics(metrics)

                logging.info("Model evaluation completed.")
                logging.info(f"Metrics: {metrics}")

        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}")
            raise

    def _load_model(self):
        """
        Loads the model from the specified path.

        Returns:
            Model: The loaded Keras model.
        """
        try:
            model_path = os.path.join(self.model_path, "model.keras")
            model = load_model(model_path)
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def _load_test_data(self):
        """
        Loads the test data from the CSV file.

        Returns:
            tuple: Features and target variable of the test dataset.
        """
        try:
            test_path = os.path.join(self.test_data_path, "test.csv")
            df_test = pd.read_csv(test_path)
            X_test = df_test.drop('Attrition', axis=1)
            y_test = df_test['Attrition']
            logging.info("Test data loaded successfully.")
            return X_test, y_test
        except Exception as e:
            logging.error(f"Error loading test data: {e}")
            raise
