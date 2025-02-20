import click
import os
from pathlib import Path
import shutil


def generate_dockerfile(
    processing_name: str, base_docker_image: str, pipeline_dir: str
):
    """
    Generates a Dockerfile for the processing pipeline.

    Args:
        processing_name (str): Name of the processing pipeline.
        base_docker_image (str): Base Docker image to use.
        pipeline_dir (str): Path to the pipeline directory where the Dockerfile will be created.
    """
    dockerfile_content = f"""FROM {base_docker_image}

RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

COPY ./dist/{processing_name} ./dist/{processing_name}

RUN  uv pip install --python=$(which python3.10) --no-cache -r ./dist/{processing_name}/requirements.txt

COPY ./examples ./examples
COPY ./src/decorators ./src/decorators
COPY ./src/models ./src/models
COPY ./src/steps ./src/steps
COPY ./src/*.py ./src

ENV PYTHONPATH=":/experiment/src"

ENTRYPOINT ["run", "python3.10", "./dist/{processing_name}/processing_pipeline.py"]
"""
    dockerfile_path = os.path.join(pipeline_dir, "Dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    click.echo(f"Dockerfile created at: {dockerfile_path}")


def copy_file(source: str, destination: str):
    """
    Copies a file from source to destination, creating directories if needed.

    Args:
        source (str): Path to the source file.
        destination (str): Path to the destination file.
    """
    destination_dir = os.path.dirname(destination)
    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source, destination)
    click.echo(f"Copied: {source} -> {destination}")


def get_repository_root() -> str:
    """
    Finds the root directory of the Git repository.

    Returns:
        str: Path to the repository root.

    Raises:
        RuntimeError: If the repository root cannot be found.
    """
    current_path = Path.cwd()
    while not (current_path / ".git").is_dir() and current_path != current_path.parent:
        current_path = current_path.parent

    if current_path == current_path.parent:
        raise RuntimeError(
            "Repository root not found. Make sure you're inside a Git repository."
        )
    return str(current_path)
