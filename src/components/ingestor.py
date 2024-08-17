"""Module for ingesting data."""

import os
import mlflow
import pandas as pd
from src.logger import logging
from src.constants import COLS_TO_DROP, VALUES_TO_REPLACE


class DataIngestor:
    """
    A class for ingesting data.

    This class provides methods for reading data from a CSV file,
    processing it, and saving it to a new location.

    Attributes:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to the output CSV file.

    Methods:
        ingest(): Reads data from the input CSV file, processes it,
        and saves it to the output CSV file.
    """

    def __init__(self, input_path: str, output_path: str):
        """
        Initializes a DataIngestor object with input and output paths.

        Parameters:
            input_path (str): The path to the input CSV file.
            output_path (str): The path to the output CSV file.
        """
        self.input_path: str = input_path
        self.output_path: str = output_path

    def ingest(self):
        """
        Ingests data from a CSV file, processes it, and logs relevant
        parameters using MLFlow.

        The function reads a CSV file from the specified input path, drops
        unnecessary columns, replaces categorical values with numerical values,
        and logs the number of rows and columns.

        It then saves the processed data to a CSV file in the specified output
        path.

        Parameters:
            None (uses instance variables input_path and output_path)

        Returns:
            None
        """

        os.makedirs(self.output_path, exist_ok=True)

        output_file = os.path.join(self.output_path, "data.csv")

        with mlflow.start_run(run_name="Ingestion Data", nested=True):
            if os.path.exists(output_file):
                logging.info("Data already exists in %s", self.output_path)
                mlflow.log_artifact(output_file)
                return

            logging.info("Reading data from %s", self.input_path)
            df = pd.read_csv(self.input_path)

            df.drop(COLS_TO_DROP, axis=1, inplace=True)

            mlflow.log_param("num_rows", df.shape[0])
            mlflow.log_param("num_cols", df.shape[1])

            df['Attrition'] = df['Attrition'].map(VALUES_TO_REPLACE)
            df['OverTime'] = df['OverTime'].map(VALUES_TO_REPLACE)

            df.to_csv(output_file, index=False)
            mlflow.log_artifact(output_file)

            logging.info("Data saved to %s", self.output_path)
