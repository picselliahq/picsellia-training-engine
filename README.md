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

If you create your own Dockerfile, its structure should look like this:

```Dockerfile
FROM picsellia/cuda:<base-tag>

COPY your-custom-image/requirements.txt .
RUN pip install -r requirements. txt

# Note: You can also put the picsellia package inside your requirements
RUN pip install -r picsellia --upgrade

WORKDIR /picsellia

COPY your-custom-image/picsellia .

ENTRYPOINT ["run", "main.py"]
```

Using `ENTRYPOINT ["run", "main.py"]` will ensure that the log container's output is automatically directed to your
Picsellia workspace.

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
