import os
import click
from cli.utils.prompt import fetch_processing_name
from cli.utils.validation import validate_and_update_processing
from cli.utils.collect_params import update_processing_parameters
from cli.utils.dockerfile_generation import (
    generate_dockerfile,
    copy_file,
    get_repository_root,
)


@click.command()
def setup_dockerized_pipeline():
    """
    Use session values to set up a dockerized pipeline.
    """
    processing_name = fetch_processing_name()
    if not processing_name:
        return

    processing = validate_and_update_processing(processing_name)
    if not processing:
        return

    base_docker_image = click.prompt(
        "Base Docker image",
        default="picsellia/cpu:python3.10",
        show_default=True,
    )

    try:
        repo_root = get_repository_root()
        pipeline_dir = os.path.join(repo_root, "dist", processing_name)
        os.makedirs(pipeline_dir, exist_ok=True)

        update_processing_parameters(
            os.path.join(repo_root, processing["picsellia_pipeline_script_path"]),
            processing["parameters"],
        )

        # Copy files
        copy_file(
            source=os.path.join(
                repo_root, processing["picsellia_pipeline_script_path"]
            ),
            destination=os.path.join(pipeline_dir, "processing_pipeline.py"),
        )
        copy_file(
            source=os.path.join(repo_root, processing["requirements_path"]),
            destination=os.path.join(pipeline_dir, "requirements.txt"),
        )

        # Generate Dockerfile
        generate_dockerfile(processing_name, base_docker_image, pipeline_dir)

        click.echo(
            f"Dockerized pipeline '{processing_name}' setup in '{pipeline_dir}'."
        )
    except Exception as e:
        click.echo(f"Error setting up dockerized pipeline: {e}")
