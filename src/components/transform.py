from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import mlflow
from src.logger import logging


class Transform:
    """
    Class for transforming data.
    """

    def __init__(self, input_path: str,
                 output_path: str):

        """
        Initializes the Transform class.

        Args:
            input_path (str): The path to the input CSV file.
            train_output_path (str): The path to the output train CSV file.
            test_output_path (str): The path to the output test CSV file.
        """

        self.input_path = os.path.join(input_path, os.listdir(input_path)[0])
        self.output_path = output_path

    def transform(self):
        """
        Transforms data from input_path by separating features into numerical
        and categorical, applying scaling and encoding, and then splitting into
        training and testing datasets.

        Raises:
        FileNotFoundError: If the CSV file at input_path does not exist.
        KeyError: If the target column 'Attrition' is not in the dataset.
        Exception: For any other unexpected issues during data transformation.
        """
        try:
            with mlflow.start_run(run_name="Data Transformation", nested=True):
                logging.info("Data Transformation")

                df = pd.read_csv(self.input_path)
                y = df["Attrition"]
                X = df.drop("Attrition", axis=1)

                X_cat = X.select_dtypes("object")
                X_num = X.drop(X_cat, axis=1)

                num_pipe = Pipeline([("scaler", StandardScaler())])
                cat_pipe = Pipeline([("onehot",
                                    OneHotEncoder(handle_unknown="ignore"))])

                preprocessor = ColumnTransformer([
                    ("num", num_pipe, X_num.columns),
                    ("cat", cat_pipe, X_cat.columns)
                ])

                X_processed = preprocessor.fit_transform(X)

                X_processed_df = pd.DataFrame(
                    X_processed,
                    columns=preprocessor.get_feature_names_out()
                )

                df_processed = pd.concat([X_processed_df,
                                          y.reset_index(drop=True)], axis=1)

                mlflow.log_param("num_trainable_features",
                                 X_processed.shape[1])
                mlflow.log_param("num_categorical_features", X_cat.shape[1])
                mlflow.log_param("num_numerical_features", X_num.shape[1])
                mlflow.log_param("col_to_classify", "Attrition")

                train_df, test_df = train_test_split(df_processed,
                                                     test_size=0.2,
                                                     random_state=42)

                os.makedirs(self.output_path, exist_ok=True)

                train_output_path = os.path.join(self.output_path, "train.csv")
                test_output_path = os.path.join(self.output_path, "test.csv")

                train_df.to_csv(train_output_path, index=False)
                test_df.to_csv(test_output_path, index=False)

                mlflow.log_artifact(train_output_path)
                mlflow.log_artifact(test_output_path)

                logging.info("Data Transformation Completed")

        except FileNotFoundError as e:
            logging.error("Input file not found: %s", e)
            raise
        except KeyError as e:
            logging.error("Missing target column in dataset: %s", e)
            raise
        except Exception as e:
            logging.error("An error occurred during data transformation: %s",
                          e)
            raise
