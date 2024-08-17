""" Class for Estatistics Generation. """

import os
import mlflow
import pandas as pd
import joblib
from src.logger import logging


class EstatisticGen:
    """
    Class for generating data statistics.
    """

    def __init__(self, input_path: str, output_path: str):
        """
        Initializes the EstatisticGen class.

        Args:
            input_path (str): The path to the input CSV file.
            output_path (str): The path to the output JSON file.
        """
        self.input_path = os.path.join(input_path, os.listdir(input_path)[0])
        self.output_path = output_path

    def generate_stats(self):
        """
        Generates a data statistics from a CSV file.
        """

        os.makedirs(self.output_path, exist_ok=True)

        output_path = os.path.join(self.output_path, "statistics.json")

        with mlflow.start_run(run_name="Estatistics Generation", nested=True):

            if os.path.exists(output_path):
                logging.info("Estatistics already exists in %s", output_path)
                mlflow.log_artifact(output_path)
                return

            logging.info("Estatistics Generation")
            df = pd.read_csv(self.input_path)
            stats = df.describe().to_dict()

            joblib.dump(stats, output_path)

            mlflow.log_dict(stats, "statistics.json")
            mlflow.log_artifact(output_path)
            logging.info("Estatistics Generation Finished")
