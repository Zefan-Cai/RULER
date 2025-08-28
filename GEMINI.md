# Project RULER: Long-Context Language Model Evaluation

## Project Overview

This project, "RULER," is a framework for evaluating the long-context capabilities of large language models (LLMs). It provides tools to generate synthetic datasets with configurable sequence lengths and task complexities, and then benchmark models against these datasets. The primary goal is to assess how well models perform on tasks requiring understanding of long sequences of text, going beyond simple "needle-in-a-haystack" recall tests.

The project is built with Python and uses a variety of libraries for interacting with LLMs, including Hugging Face `transformers`, `vllm`, and APIs for OpenAI and Google Gemini. The evaluation environment is managed using Docker.

## Building and Running

The project uses a shell script-based workflow for running evaluations.

### 1. Environment Setup

The project is designed to run within a Docker container. You can build the container using the provided Dockerfile:

```bash
cd docker/
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t cphsieh/ruler:0.2.0 .
```

### 2. Data Download

Before running an evaluation, you need to download the necessary datasets:

```bash
cd scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
```

### 3. Running an Evaluation

To run an evaluation, use the `run.sh` script, specifying the model and benchmark to use.

**Example:**

```bash
bash run.sh llama2-7b-chat synthetic
```

**Configuration:**

*   **`scripts/run.sh`**: This is the main script. You may need to edit the `ROOT_DIR`, `MODEL_DIR`, and `ENGINE_DIR` variables to match your local setup.
*   **`scripts/config_models.sh`**: This file contains the configurations for the models you want to evaluate. You can add new models or modify existing ones here. You will also need to add your API keys for services like OpenAI and Gemini in this file.
*   **`scripts/config_tasks.sh`**: This file defines the tasks that are part of a benchmark.
*   **`scripts/synthetic.yaml`**: This file contains the configuration for the synthetic data generation, allowing you to control the complexity of the tasks.

## Development Conventions

*   **Model Support:** To add a new model, you need to:
    1.  Add a new case to the `MODEL_SELECT` function in `scripts/config_models.sh`.
    2.  Define the `MODEL_PATH`, `MODEL_TEMPLATE_TYPE`, and `MODEL_FRAMEWORK`.
    3.  If the model uses a new chat template, add it to `scripts/data/template.py`.
*   **Task Contribution:** To contribute a new synthetic task, you need to:
    1.  Create a Python script for data preparation in `scripts/data/synthetic`.
    2.  Add a task template to `scripts/data/synthetic/constants.py`.
    3.  Add an evaluation metric to `scripts/eval/synthetic/constants.py`.
    4.  Define the task in `scripts/synthetic.yaml` and `scripts/config_tasks.sh`.
