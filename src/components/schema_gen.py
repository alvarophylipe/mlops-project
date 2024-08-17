""" Module for generating data schema. """

import os
from copy import copy
from typing import List
import pandera as pa
import pandas as pd
import mlflow
from src.logger import logging


class SchemaGen:
    """
    Class for generating data schema from a CSV file.
    """

    def __init__(self, input_path: str, output_path: str) -> None:
        """
        Initializes the SchemaGen class.

        Args:
            input_path (str): The path to the input CSV file.
            output_path (str): The path to the output JSON file.
        """
        self.input_path = os.path.join(input_path, os.listdir(input_path)[0])
        self.output_path = output_path

    def generate_schema(self) -> None:
        """
        Generates a data schema from a CSV file.
        """

        os.makedirs(self.output_path, exist_ok=True)

        output_file = os.path.join(self.output_path, "schema.json")

        with mlflow.start_run(run_name="Data Schema Generation", nested=True):

            if os.path.exists(output_file):
                logging.info("Data schema already exists in %s", output_file)
                mlflow.log_artifact(output_file)
                return

            logging.info("Generating data schema from %s", self.input_path)
            df = pd.read_csv(self.input_path)

            schema_json = self._infer_schema(df)
            loaded_schema = self._load_schema_from_json(schema_json)
            modified_schema = self._modify_schema_based_on_df(loaded_schema,
                                                              df)
            if self.validate_dataframe(df, modified_schema):
                modified_schema.to_json(output_file)
                logging.info("Data schema saved to %s", output_file)
                mlflow.log_artifact(output_file)
                logging.info("Data schema generation completed successfully")
            else:
                raise ValueError("Data schema not generated successfully")

    def _infer_schema(self, df: pd.DataFrame) -> str:
        """
        Infers the data schema from a CSV file.

        Args:
            df (pd.DataFrame): The dataframe from which to infer the schema.

        Returns:
            str: The inferred schema in JSON format.
        """

        logging.info("Inferring data schema")
        schema = pa.infer_schema(df)
        logging.info("Data schema inferred")
        return schema.to_json()

    def _load_schema_from_json(self, schema_json: str) -> pa.DataFrameSchema:
        """
        Loads a data schema from a JSON representation.

        Args:
            schema_json (str): The JSON representation of the schema.

        Returns:
            pa.DataFrameSchema: The loaded data schema.
        """
        logging.info("Loading data schema from JSON")
        loaded_schema = pa.DataFrameSchema.from_json(schema_json)
        logging.info("Data schema loaded")
        return loaded_schema

    def _modify_schema_based_on_df(self, schema: pa.DataFrameSchema,
                                   df: pd.DataFrame) -> pa.DataFrameSchema:
        """
        Modifies a data schema based on the dataframe provided.

        Args:
            schema (pa.DataFrameSchema): The schema to modify.
            df (pd.DataFrame): The dataframe containing the data.

        Returns:
            pa.DataFrameSchema: The modified schema.
        """
        modified_schema = copy(schema)
        for col in df.columns:
            column_dtype = df[col].dtype

            if column_dtype == 'object':
                checks = [pa.Check.isin(list(df[col].unique()))]
            else:
                checks = [
                    pa.Check.greater_than_or_equal_to(int(0)),
                    pa.Check.less_than_or_equal_to(int(df[col].max()))
                ]

            modified_schema = self._modify_schema(modified_schema, col, checks)
        return modified_schema

    def _modify_schema(self, schema: pa.DataFrameSchema,
                       column: str,
                       checks: List[pa.Check]) -> pa.DataFrameSchema:
        """
        Modifies a data schema for a specific column.

        Args:
            schema (pa.DataFrameSchema): The schema to modify.
            column (str): The column to modify.
            checks (List[pa.Check]): The checks to apply to the column.

        Returns:
            pa.DataFrameSchema: The modified schema.
        """
        if column in schema.columns:
            logging.info("Modifying data schema for column %s", column)
            schema.columns[column].checks = checks
            logging.info("Data schema modified for column %s", column)
        return schema

    def validate_dataframe(self, df: pd.DataFrame,
                           schema: pa.DataFrameSchema) -> bool:
        """
        Validates a dataframe against a data schema.

        Args:
            df (pd.DataFrame): The dataframe to validate.
            schema (pa.DataFrameSchema): The schema to validate against.

        Returns:
            bool: True if the dataframe is valid, False otherwise.
        """
        try:
            logging.info("Validating dataframe against the schema.")
            schema.validate(df)
            logging.info("Dataframe is valid.")
            return True
        except pa.errors.SchemaError as e:
            logging.error("Validation error: %s", e)
            return False
