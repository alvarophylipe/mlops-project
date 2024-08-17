"""Constants for the project."""

from typing import List, Dict
import os

# General Constants
SRC_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT: str = os.path.dirname(SRC_PATH)
ARTIFACTS_PATH: str = os.path.join(PROJECT_ROOT, "artifacts")
NOTEBOOK_PATH: str = os.path.join(PROJECT_ROOT, "notebook")

# Data Ingestion Constants
DATA_INGESTION_INPUT_PATH: str = os.path.join(NOTEBOOK_PATH, "data",
                                              "IBM_Employee_Attrition.csv")
DATA_INGESTION_OUTPUT_PATH: str = os.path.join(ARTIFACTS_PATH, "raw")
COLS_TO_DROP: List[str] = ["EmployeeCount", "EmployeeNumber",
                           "Over18", "StandardHours"]
VALUES_TO_REPLACE: Dict[str, int] = {"Yes": 1, "No": 0}

# Statistics Generation Constants
STATS_GEN_OUTPUT_PATH: str = os.path.join(ARTIFACTS_PATH, "statistics")

# Schema Generation Constants
SCHEMA_GEN_OUTPUT_PATH: str = os.path.join(ARTIFACTS_PATH, "schema")

# Transformation Constants
TRANSFORM_OUTPUT_PATH: str = os.path.join(ARTIFACTS_PATH, "processed")

# Model Training Constants
MODEL_TRAINING_OUTPUT_PATH: str = os.path.join(
    PROJECT_ROOT, "models", "model"
)
FIRST_DENSE_LAYER_UNITS: int = 48
SECOND_DENSE_LAYER_UNITS: int = 32
OUTPUT_DENSE_LAYER_UNITS: int = 1
FIRST_DROPOUT_LAYER: float = 0.2
SECOND_DROPOUT_LAYER: float = 0.3
RELU_ACTIVATION: str = "relu"
SIGMOID_ACTIVATION: str = "sigmoid"
EPOCHS: int = 50
BATCH_SIZE: int = 32
COMPILE_ARGS = {
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"],
}
EARLY_STOPPING_ARGS = {
    "monitor": "val_loss",
    "patience": 5,
    "restore_best_weights": True
}


# Model Evaluation Constants
EVALUATION_OUTPUT_PATH: str = os.path.join(ARTIFACTS_PATH, "evaluation")

# Pusher Constants
PUSHER_OUTPUT_PATH: str = os.path.join(
    PROJECT_ROOT, "models", "pushed"
)
