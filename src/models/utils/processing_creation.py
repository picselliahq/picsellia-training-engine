import shutil
import os
from picsellia import Client

from picsellia.types.enums import ProcessingType


def find_repository_root():
    """
    Identifies and returns the root directory of the repository by looking for the `examples` folder.

    Returns:
        str: The root directory path of the repository.

    Raises:
        RuntimeError: If the `examples` directory is not found in the current path.
    """
    current_path = os.getcwd()
    if "examples" in current_path:
        return current_path.split("examples")[0].rstrip("/")
    else:
        raise RuntimeError(
            "Unable to locate 'examples' in the current path to determine the repository root."
        )


def create_picsellia_processing_pipeline(
    api_token: str,
    organization_id: str,
    processing_name: str,
    default_parameters: dict,
    docker_image: str,
    docker_tag: str,
    processing_type: ProcessingType,
):
    """
    Configures and registers a new processing pipeline in Picsellia.

    Args:
        api_token (str): API token for authentication.
        organization_id (str): Organization ID in Picsellia.
        processing_name (str): Name of the processing pipeline.
        default_parameters (dict): Default parameters for the processing pipeline.
        docker_image (str): Docker image name.
        docker_tag (str): Docker image tag.
        processing_type (ProcessingType): Type of processing (e.g., DATASET_VERSION_CREATION).
    """

    client = Client(api_token=api_token, organization_id=organization_id)

    client.create_processing(
        name=processing_name,
        type=processing_type,
        default_cpu=4,
        default_gpu=0,
        default_parameters=default_parameters,
        docker_image=docker_image,
        docker_tag=docker_tag,
        docker_flags=None,
    )


def generate_pipeline_dockerfile(
    pipeline_directory: str,
    processing_name: str,
    base_docker_image: str = "picsellia/cpu:python3.10",
):
    """
    Creates a Dockerfile for the specified processing pipeline.

    Args:
        pipeline_directory (str): Path to the pipeline directory where the Dockerfile will be created.
        processing_name (str): Name of the processing pipeline.
        base_docker_image (str): Base Docker image to use (default is "picsellia/cpu:python3.10").
    """
    dockerfile_content = f"""FROM {base_docker_image}

RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    && rm -rf /var/lib/apt/lists/*

COPY ./src/pipelines/{processing_name}/requirements.txt ./src/pipelines/{processing_name}/requirements.txt

RUN  uv pip install --python=$(which python3.10) --no-cache -r ./src/pipelines/{processing_name}/requirements.txt

WORKDIR /experiment

COPY ./examples ./examples
COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src
COPY ./src/pipelines/{processing_name} ./src/pipelines/{processing_name}

ENV PYTHONPATH=":/experiment/src"

ENTRYPOINT ["run", "python3.10", "src/pipelines/{processing_name}/processing_pipeline.py"]
"""
    with open(f"{pipeline_directory}/Dockerfile", "w") as f:
        f.write(dockerfile_content)

    print(f"Dockerized pipeline '{processing_name}' created in {pipeline_directory}/")


def copy_file_to_destination(source_path, destination_path):
    """
    Copies a file from the source path to the destination path.

    Args:
        source_path (str): Path to the source file.
        destination_path (str): Path to the destination directory or file.

    Raises:
        FileNotFoundError: If the source file does not exist.
    """
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
    else:
        raise FileNotFoundError(f"The file was not found: {source_path}")


def setup_dockerized_pipeline(
    api_token: str,
    organization_id: str,
    processing_name: str,
    pipeline_script_path: str,
    requirements_file_path: str,
    docker_image: str,
    docker_tag: str,
    default_parameters: dict,
    base_docker_image: str = "picsellia/cpu:python3.10",
):
    """
    Automates the setup of a dockerized processing pipeline, including copying files, generating a Dockerfile,
    and registering the pipeline in Picsellia.

    Args:
        api_token (str): API token for Picsellia.
        organization_id (str): Organization ID in Picsellia.
        processing_name (str): Name of the processing pipeline.
        pipeline_script_path (str): Path to the Python script for the processing pipeline.
        requirements_file_path (str): Path to the requirements.txt file.
        docker_image (str): Docker image name.
        docker_tag (str): Docker image tag.
        default_parameters (dict): Default parameters for the processing pipeline.
        base_docker_image (str): Base Docker image to use (default is "picsellia/cpu:python3.10").
    """
    # Identify the repository root
    repo_root = find_repository_root()

    # Define the pipeline directory
    pipeline_directory = os.path.join(repo_root, f"src/pipelines/{processing_name}")
    os.makedirs(pipeline_directory, exist_ok=True)

    # Copy the processing pipeline script and rename it to `processing_pipeline.py`
    copy_file_to_destination(
        source_path=pipeline_script_path,
        destination_path=os.path.join(pipeline_directory, "processing_pipeline.py"),
    )

    # Copy the requirements.txt file
    copy_file_to_destination(
        source_path=requirements_file_path,
        destination_path=os.path.join(pipeline_directory, "requirements.txt"),
    )

    # Generate the Dockerfile
    generate_pipeline_dockerfile(
        pipeline_directory=pipeline_directory,
        processing_name=processing_name,
        base_docker_image=base_docker_image,
    )

    # Register the processing pipeline in Picsellia
    create_picsellia_processing_pipeline(
        api_token=api_token,
        organization_id=organization_id,
        processing_name=processing_name,
        default_parameters=default_parameters,
        docker_image=docker_image,
        docker_tag=docker_tag,
        processing_type=ProcessingType.DATASET_VERSION_CREATION,
    )
