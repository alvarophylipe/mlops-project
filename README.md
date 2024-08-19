# Pipeline de Previsão de Evasão de Funcionários

Este repositório contém um pipeline de machine learning para prever a evasão de funcionários. O pipeline lida com ingestão de dados, transformação, treinamento de modelo, avaliação e implantação, utilizando diversas bibliotecas Python, incluindo `pandas`, `scikit-learn`, `tensorflow` e `mlflow`.

## Índice

- [Visão Geral do Projeto](#visão-geral-do-projeto)
- [Funcionalidades](#funcionalidades)
- [Instalação](#instalação)
- [Uso](#uso)
  - [Executando o Pipeline](#executando-o-pipeline)
  - [Executando com Argumentos de Linha de Comando](#executando-com-argumentos-de-linha-de-comando)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Licença](#licença)

## Visão Geral do Projeto

O pipeline foi projetado para automatizar o processo de desenvolvimento e implantação de um modelo de machine learning para prever a evasão de funcionários. Ele inclui os seguintes componentes:

- **Ingestão de Dados**: Lê e processa dados a partir de arquivos CSV.
- **Transformação de Dados**: Separa os dados em conjuntos de treino e teste e aplica escalonamento e codificação às features.
- **Treinamento de Modelo**: Treina um modelo de rede neural usando os dados pré-processados.
- **Avaliação de Modelo**: Avalia o modelo treinado e registra métricas no MLflow.
- **Implantação de Modelo**: Prepara o modelo treinado para implantação em futuras previsões.

## Funcionalidades

- Ingestão e pré-processamento de dados automatizados.
- Treinamento e avaliação de modelo com TensorFlow.
- Integração com MLflow para rastreamento de experimentos e gestão de modelos.
- Design flexível e modular, permitindo fácil customização e extensão.

## Instalação

Para configurar o projeto, siga estes passos:

1. **Clone o repositório**:
```bash
git clone https://github.com/alvarophylipe/mlops-project.git
cd mlops-project
```

2. **Delete a pasta artifacts**:
```bash
rmdir artifacts
```

3. **Crie um ambiente virtual**:
```bash
python3 -m venv venv
source venv/bin/activate   # No Windows use `venv\Scripts\activate`
```

4. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

5. **Configure o MLflow** (opcional, mas recomendado):
```bash
mlflow server
```
## Uso

### Executando o Pipeline

Você pode executar todo o pipeline utilizando o script Python fornecido:

```bash
python main.py --experiment_name "Previsão de Evasão de Funcionários" --csv_path <datafile_path>
```

## Executando com Argumentos de Linha de Comando

Você pode personalizar a execução passando argumentos de linha de comando:

```bash
python main.py --experiment_name "Previsão de Evasão de Funcionários" --csv_path <datafile_path> --tracking_uri "http://localhost:5000" --artifact_location "mlruns"
```
## Exemplo de Uso em Script

Aqui está um exemplo de como usar o pipeline dentro de um script:

```py
from src.pipeline.pipeline_training import run_pipeline
from src.utils import create_mlflow_experiment
from src.logger import logging
import warnings
import mlflow

warnings.filterwarnings("ignore")

logging.getLogger("mlflow").setLevel(logging.ERROR)

if __name__ == "__main__":

    create_mlflow_experiment(
        experiment_name="Previsão de Evasão de Funcionários",
        artifact_location="mlruns",
        tags={"env": "dev", "version": "1.0.0"}
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    run_pipeline("Previsão de Evasão de Funcionários", "data/data.csv")
```
Este script configura o URI de rastreamento do MLflow e executa o pipeline sob o nome do experimento especificado.

## Estrutura de Diretórios

```bash
├── artifacts/                      # Diretório de armazenamento de artefatos
├── mlruns/                         # Diretório de rastreamento de experimentos do MLflow
├── src/                            # Código-fonte do pipeline
│   ├── components/                 # Ingestão de dados, transformação, etc.
│   ├── constants/                  # Variáveis constantes
│   ├── pipeline/                   # Scripts do pipeline
│   ├── utils/                      # Funções utilitárias
│   └── logger.py                   # Configurações de logging
├── README.md                       # README do projeto
├── LICENSE                         # Licença do projeto
├── requirements.txt                # Dependências Python
└── main.py                         # Script para rodar o pipeline de ML
```

## Licença

Este projeto está licenciado sob os termos da GNU General Public License (GPL) v3.0. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.