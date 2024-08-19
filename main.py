import subprocess
import argparse
import warnings
import mlflow
from src.utils import create_mlflow_experiment
from src.pipeline.pipeline_training import run_pipeline
from src.logger import logging

warnings.filterwarnings("ignore")

logging.getLogger("mlflow").setLevel(logging.ERROR)


def start_mlflow_ui(port=5000):
    """
    Inicia o MLflow UI em um subprocesso.

    Args:
        port (int): Porta em que o MLflow UI será executado.
    """
    subprocess.Popen(["mlflow", "ui", "--port", str(port)])
    logging.info(f"MLflow UI started at http://localhost:{port}")


def main(experiment_name, csv_path, tracking_uri="http://localhost:5000",
         artifact_location="mlruns", port=5000, env="prod", version="1.0.0"):
    """
    Função principal para rodar a pipeline de treinamento de ML.

    Args:
        experiment_name (str): Nome do experimento no MLflow.
        tracking_uri (str): URI do servidor de rastreamento MLflow.
        artifact_location (str): Local de armazenamento dos artefatos do
        MLflow.
    """

    create_mlflow_experiment(
        experiment_name=experiment_name,
        artifact_location=artifact_location,
        tags={"env": env, "version": version}
    )

    logging.getLogger("mlflow").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(tracking_uri)

    start_mlflow_ui(port=port)  # Inicia o MLflow UI na porta 5000

    run_pipeline(experiment_name, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executar a pipeline de ML")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Nome do experimento MLflow")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Caminho do arquivo CSV")
    parser.add_argument("--tracking_uri", type=str,
                        default="http://localhost:5000",
                        help="URI do servidor MLflow")
    parser.add_argument("--artifact_location", type=str, default="mlruns",
                        help="Local de armazenamento dos artefatos MLflow")
    parser.add_argument("--port", type=int, default=5000,
                        help="Porta do MLflow UI")
    parser.add_argument("--env", type=str, default="prod", help="Ambiente")
    parser.add_argument("--version", type=str, default="1.0.0", help="Versão")

    args = parser.parse_args()

    main(experiment_name=args.experiment_name, csv_path=args.csv_path,
         tracking_uri=args.tracking_uri,
         artifact_location=args.artifact_location, port=args.port,
         env=args.env, version=args.version)
