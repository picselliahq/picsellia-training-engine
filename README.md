<h1 align="center">
    <img margin="0 10px 0 0" src="https://www.docker.com/wp-content/uploads/2022/03/vertical-logo-monochromatic.png" width="150px"/>
    <img margin="0 10px 0 0" src="https://uploads-ssl.webflow.com/60d1a7f5aeb33cb8af546898/610bcf4cc7ae73979fa0d23b_256.png" width="120px"/>
</h1>
  <h2 align="center">Picsellia Public Repo</h2>
  <p align="center">A collection of Docker images provided for your trainings!
<p>
<p align="center">
    <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-brightgreen.svg" alt="Python supported"/></a>
</p>

The Picsellia Training Engine is a project designed to provide an efficient, customizable, and Docker-based system for
training ML models. Its main purpose is to interact seamlessly with the Picsellia SAS platform,
leveraging the functionalities and capabilities it provides.

The repository provides images for the following ML Frameworks:

- Tensorflow 2
- Keras
- ONNX

# Installation ⚙️

You'll just need to install [Docker](https://docs.docker.com/engine/install/), then you can:

- Select an image from this repository and modify it to suit your needs. Please ensure you follow
  our [specific guidelines](https://dash.readme.com/project/picsellia-docs/v2.2/docs/integrate-picsellia-into-your-training-scripts)
  to ensure compatibility with Picsellia.
- Create your own Dockerfile and training script. For this, you can use one of our base images. Pull it directly from
  Docker Hub using `picsellia/cuda:<base-tag>`. The available tags
  are [here](https://hub.docker.com/r/picsellia/cuda/tags).


# UV Development Environment Setup 🚀

Streamline your Python data science projects with `uv`. This guide covers setup instructions from installation to dependency management.

## Dependency Files Explained

Before diving in, let's clarify the roles of `requirements.in` and `requirements.txt`:

- **`requirements.in`:** Specify your project's direct dependencies and their version ranges here for flexibility and compatibility.
- **`requirements.txt`:** Generated from `requirements.in` by `uv`, this file pins all dependencies to specific versions for a consistent and stable environment.

## Setup Instructions

1. **Python Installation:** Ensure Python (versions 3.8 to 3.11 supported) is installed on your system.


2. **Install `uv`:** Pick the right `uv` installation method for your OS.
   - **macOS and Linux:**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - **Windows:**
     ```bash
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - **With pip:**
     ```bash
     pip install uv
     ```
   - **With Homebrew:**
      ```bash
      brew install uv
      ```

3. **Navigate to your project directory:** (for example `yolov8-detection`, `tf2` or a new project directory)

    ```bash
    cd yolov8-detection
    ```

4. **Create and activate a virtual environment:** With `uv venv` a virtual environment is created in the `.venv` directory.

    ```bash
    uv venv
    source .venv/bin/activate
    ```

5. **(Optional) Increase HTTP Timeout:** To prevent timeouts during package installation (step 6.), especially for larger packages like PyTorch, increase the HTTP timeout:

    ```bash
    export UV_HTTP_TIMEOUT=600
    ```


## Using an Existing Model

If you're working with an existing model that includes both `requirements.in` and `requirements.txt`, follow these steps:


1. **Install dependencies:** Install the required dependencies from the requirements.txt file:

    ```bash
    uv pip install -r requirements.txt
    ```

### Optional Steps:

- **Upgrade Dependencies:** To update dependencies based on requirements.in, recompile to generate an updated requirements.txt:

    ```bash
    uv pip compile requirements.in -o requirements.txt
    ```

- **Adding a New Package:** If you need to add a new package, insert it into requirements.in with version constraints, then recompile to update requirements.txt:

    ```bash
    echo "flask>=2.0.0, <3.0" >> requirements.in
    uv pip compile requirements.in -o requirements.txt
    ```

## Creating a New Model

When starting a new model, you'll need to create `requirements.in` and generate `requirements.txt` to manage dependencies. Follow these steps to set up your project:

1. **Define Dependencies:** Create requirements.in with your model's dependencies. Ensure to specify versions that are tested and compatible with your model.

    - For a range of acceptable versions (e.g., Flask):
        ```bash
        flask>=2.0.0, <3.0
        ```
    - For a fixed version when compatibility is crucial:
        ```bash
        flask==2.0.0
        ```
    - For specifying a minimum version:
        ```bash
        flask>=2.0.0
        ```

2. **Lock Dependencies:** Generate requirements.txt from your requirements.in.

    ```bash
    uv pip compile requirements.in -o requirements.txt
    ```

3. **Install Dependencies:** Proceed with installing dependencies specified in requirements.txt.

    ```bash
    uv pip install -r requirements.txt
    ```

# Configuration 🛠️️

This section guides you through setting up and customizing your Docker environment using Picsellia Docker images. You can choose from a variety of CPU and CUDA-based images depending on your specific requirements for Python and CUDA versions.

## Selecting the Base Image

Picsellia provides a range of Docker images suitable for various machine learning tasks. These images are pre-configured with essential libraries and Python, making them an starting point for your projects.

### CPU-based Images
These images are optimized for systems without a GPU and are suitable for less computationally intensive tasks or for development and testing purposes:

- **Python 3.8**: picsellia/cpu:ubuntu20.04-python3.8
- **Python 3.9**: picsellia/cpu:ubuntu20.04-python3.9
- **Python 3.10**: picsellia/cpu:ubuntu20.04-python3.10
- **Python 3.11**: picsellia/cpu:ubuntu20.04-python3.11
- **Python 3.12**: picsellia/cpu:ubuntu20.04-python3.12

### CUDA-based Images
These images are equipped with CUDA and cuDNN libraries, making them ideal for GPU-accelerated machine learning tasks:

- **CUDA 11.4.3 with cuDNN 8**:
  - **Python 3.8**: picsellia/cuda:11.4.3-cudnn8-ubuntu20.04-python3.8

- **CUDA 11.7.1 with cuDNN 8**:
  - **Python 3.10**: picsellia/cuda:11.7.1-cudnn8-ubuntu20.04-python3.10

- **CUDA 11.8.0 with cuDNN 8**:
  - **Python 3.9**: picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.9
  - **Python 3.10**: picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.10
  - **Python 3.11**: picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.11
  - **Python 3.12**: picsellia/cuda:11.8.0-cudnn8-ubuntu20.04-python3.12


## Customizing the Dockerfile

Below is a template Dockerfile adjusted to use with Picsellia’s provided images, suitable for a bounding box cropping task:

```Dockerfile
# Select the base image according to your CUDA and Python requirements
FROM picsellia/cpu:ubuntu20.04-python3.10

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Argument to trigger rebuilds
ARG REBUILD_ALL
ARG REBUILD_PICSELLIA

# Copy the requirements file and install dependencies using uv
COPY ./src/pipelines/bounding_box_cropper/requirements.txt ./src/pipelines/bounding_box_cropper/
RUN uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/bounding_box_cropper/requirements.txt

# Set the working directory
WORKDIR /experiment

# Copy the source code
COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src

# Copy the pipeline code
COPY ./src/pipelines/bounding_box_cropper ./src/pipelines/bounding_box_cropper

# Set environment variable for Python path
ENV PYTHONPATH "${PYTHONPATH}:/experiment/src"

# Specify the entry point for running the pipeline
ENTRYPOINT ["run", "python3.10", "src/pipelines/bounding_box_cropper/processing_pipeline.py"]
```

## Usage

To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:

```bash
docker build -t bounding-box-cropper .
```

To push the image to a container registry, you can use the following command:

```bash
docker push bounding-box-cropper
```


# Tests 🧪

## Pytest integration

We utilize the `pytest` framework for all our testing needs.
You can install it using the following command:

```bash
pip install pytest
```

To run all tests in the repository, you can execute the shell script that wraps our pytest commands:

```bash
bash run_all_pytest_tests.sh
```

## Simulating GitHub Actions locally

To replicate GitHub Actions locally, we leverage `act`, a tool that simulates GitHub Actions on your local machine. This allows you to test workflows before committing changes to the repository.

1 - Install `act`:

First, install act using the following command. This script will download and install the latest version of act:

```bash
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

2 - Prepare the environment file:

Create a .env file in the root directory of your repository with the necessary environment variables. Replace <votre_token_picsellia> with your actual Picsellia test token:

```bash
PICSELLIA_TEST_TOKEN=<votre_token_picsellia>
PICSELLIA_TEST_HOST=https://staging.picsellia.com/
```
Important: Do not include CODECOV_TOKEN in the .env file unless you intend to upload coverage results to Codecov during local testing.

3 - List available jobs:

Before running the actions, you may want to see a list of all available jobs defined in your GitHub Actions workflows:

```bash
act -l
```

4 - Run a specific job:

To execute a specific job, such as common-model-tests from the common_model_tests.yaml workflow, use the following command:

```bash
act -j common-model-tests
```


# Contributing 🤝

We welcome contributions from the community! While we're still working on establishing a full set of guidelines, we
encourage you to adhere to the general principles of respect for the original project, clarity in any changes you make,
and supporting explanations for your contributions.

# Research

# License 📄

This project is licensed under the MIT License. For more information, please refer to the LICENSE file in the
repository.

# Contact Information 🥑

Should you have any questions or if you want to contribute, please don't hesitate to contact us.

You can reach us:

- On our website https://picsellia.com/contact.
- By email at: [support@picsellia.com](mailto:support@picsellia.com).
- On Github: https://github.com/PN-picsell ☕
