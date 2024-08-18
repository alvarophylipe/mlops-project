"""
This module provides utility functions for the project.
"""

from typing import Any, Dict
import mlflow


def create_mlflow_experiment(experiment_name: str,
                             artifact_location: str,
                             tags: Dict[str, Any]) -> str:
    """
    Creates a new MLFlow experiment with the given name, artifact location,
    and tags.

    Args:
        experiment_name (str): The name of the experiment to be created.
        artifact_location (str): The location where artifacts will be stored.
        tags (Dict[str, Any]): A dictionary of tags to be associated with the
        experiment.

    Returns:
        str: The ID of the created experiment.
    """

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags=tags
        )

    except Exception:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name)\
            .experiment_id

        return experiment_id


def get_mlflow_experiment(experiment_id: str | None = None,
                          experiment_name: str | None = None) -> \
                            mlflow.entities.Experiment:
    """
    Gets the MLFlow experiment with the given ID or name.

    Args:
        experiment_id (str): The ID of the experiment.
        experiment_name (str): The name of the experiment.

    Returns:
        mlflow.entities.Experiment: The experiment object.
    """

    if experiment_id is not None:
        return mlflow.get_experiment(experiment_id)

    if experiment_name is not None:
        return mlflow.get_experiment_by_name(experiment_name)

    raise ValueError(
        "Either experiment_id or experiment_name must be provided."
        )
