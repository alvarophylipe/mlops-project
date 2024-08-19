import mlflow
from src.components.ingestor import DataIngestor
from src.components.estatistic_gen import EstatisticGen
from src.components.schema_gen import SchemaGen
from src.components.transform import Transform
from src.components.trainer import Trainer
from src.components.evaluator import Evaluator
from src.components.pusher import Pusher
from src.constants import (DATA_INGESTION_OUTPUT_PATH,
                           STATS_GEN_OUTPUT_PATH,
                           SCHEMA_GEN_OUTPUT_PATH,
                           TRANSFORM_OUTPUT_PATH,
                           MODEL_TRAINING_OUTPUT_PATH,
                           EVALUATION_OUTPUT_PATH,
                           PUSHER_OUTPUT_PATH)


def run_pipeline(experiment_name: str, csv_path: str) -> None:
    """
    This function runs a machine learning pipeline.

    It takes an experiment name as input and sets up the experiment using
    mlflow. It then starts a new run and performs the following tasks:
    - Data ingestion
    - Statistical analysis
    - Schema generation
    - Data transformation
    - Model training
    - Model evaluation
    - Model pushing

    The function does not return any value.

    Parameters:
    experiment_name (str): The name of the experiment.

    Returns:
    None
    """

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Pipeline"):

        ingestor = DataIngestor(
            input_path=csv_path,
            output_path=DATA_INGESTION_OUTPUT_PATH
        )
        ingestor.ingest()

        stats_gen = EstatisticGen(
            input_path=DATA_INGESTION_OUTPUT_PATH,
            output_path=STATS_GEN_OUTPUT_PATH
        )

        stats_gen.generate_stats()

        schema_gen = SchemaGen(
            input_path=DATA_INGESTION_OUTPUT_PATH,
            output_path=SCHEMA_GEN_OUTPUT_PATH
        )
        schema_gen.generate_schema()

        transform = Transform(
            input_path=DATA_INGESTION_OUTPUT_PATH,
            output_path=TRANSFORM_OUTPUT_PATH,
        )

        transform.transform()

        trainer = Trainer(
            input_path=TRANSFORM_OUTPUT_PATH,
            model_path=MODEL_TRAINING_OUTPUT_PATH
        )

        trainer.train()

        evaluator = Evaluator(
            model_path=MODEL_TRAINING_OUTPUT_PATH,
            test_data_path=TRANSFORM_OUTPUT_PATH,
            output_path=EVALUATION_OUTPUT_PATH
        )

        evaluator.evaluate()

        pusher = Pusher(
            model_path=MODEL_TRAINING_OUTPUT_PATH,
            push_path=PUSHER_OUTPUT_PATH
        )

        pusher.push()
