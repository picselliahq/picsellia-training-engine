import os
import subprocess
import click
from cli.helpers.prompt import fetch_processing_name
from cli.helpers.validation import validate_and_update_processing
from cli.utils.dockerfile_generation import get_repository_root


@click.command()
def build_and_push_processing():
    """
    Build and push a Docker image for a processing pipeline.
    """
    # Fetch processing name
    processing_name = fetch_processing_name()
    if not processing_name:
        return

    # Validate and update processing details
    processing = validate_and_update_processing(processing_name)
    if not processing:
        return

    # Docker image details
    image_name = processing.get("image_name")
    image_tag = processing.get("image_tag")
    full_image_name = f"{image_name}:{image_tag}"

    # Run Docker login
    click.echo("Logging into Docker. Please enter your credentials:")
    try:
        subprocess.run(["docker", "login"], check=True)
    except subprocess.CalledProcessError:
        click.echo("Docker login failed. Please try again.")
        return

    # Verify the required files in the dist directory
    try:
        repo_root = get_repository_root()
        pipeline_dir = os.path.join("dist", processing_name)

        if not os.path.exists(pipeline_dir):
            click.echo(
                f"The directory '{pipeline_dir}' does not exist. Please run the `setup-dockerized-pipeline` command first."
            )
            return

        dockerfile_path = os.path.join(pipeline_dir, "Dockerfile")
        if not os.path.exists(dockerfile_path):
            click.echo(
                f"The Dockerfile is missing in '{pipeline_dir}'. Please run the `setup-dockerized-pipeline` command first."
            )
            return

        # Build the Docker image
        click.echo(f"Building Docker image '{full_image_name}'...")
        subprocess.run(
            ["docker", "build", "-t", full_image_name, "-f", dockerfile_path, "."],
            cwd=repo_root,
            check=True,
        )

        # Push the Docker image
        click.echo(f"Pushing Docker image '{full_image_name}'...")
        subprocess.run(["docker", "push", full_image_name], check=True)

        click.echo(f"Docker image '{full_image_name}' pushed successfully!")

    except FileNotFoundError as e:
        click.echo(f"File error: {e}")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error during Docker operation: {e}")
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
