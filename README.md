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

# Development Setup 🚀

For local development, especially when working with Python and needing to manage data science dependencies, we use `uv`. Here's how you can set up your local environment to start developing:

1. **Python Installation:** Make sure you have Python installed on your system. The Picsellia Training Engine supports Python 3.8, 3.9, 3.10 and 3.11.


2. **Install `uv`:** There are several methods to install `uv`, depending on your operating system.
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

3. **Navigate to your project directory:** (for example, `yolov8-detection` or any other project like `tf2`, `unet-instance-segmentation`, etc.)

    ```bash
    cd yolov8-detection
    ```

4. **Create and activate a virtual environment:** With `uv venv` a virtual environment is created in the `.venv` directory.

    ```bash
    uv venv
    source .venv/bin/activate
    ```

5. **Increase HTTP Timeout:** To prevent timeouts during package installation, especially for larger packages like PyTorch, increase the HTTP timeout:

    ```bash
    export UV_HTTP_TIMEOUT=600
    ```

6. **Install dependencies:** Install the required dependencies from the requirements.txt file:

    ```bash
    uv pip install -r requirements.txt
    ```

7. **Managing dependency versions:** Whether you're setting up a new environment or updating an existing one with additional dependencies, the starting point always involves `requirements.in`. This file serves as the blueprint for specifying the version constraints for each package, which `uv` then uses to resolve and lock down dependencies in `requirements.txt`. This ensures your project's dependency management is both flexible and stable. Examples of how to specify dependencies in `requirements.in` include:
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
    It's crucial to validate that your model performs optimally across the specified dependency ranges by testing thoroughly. Should any version range introduce compatibility issues, it may be necessary to narrow down the specifications to a single, stable version that ensures consistent performance.


8. **Creating or updating `requirements.txt` from `requirements.in`:** Whether you're setting up `requirements.txt` for the first time or updating it to reflect changes in `requirements.in`, the command remains the same:
    ```bash
    uv pip compile requirements.in -o requirements.txt
    ```
    This command generates or updates `requirements.txt`, locking your project to specific versions of dependencies based on the guidelines you've set in `requirements.in`. This ensures your environment is both reproducible and consistent, whether you're adding new dependencies or updating existing ones.

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
