import click

from cli.utils.collect_params import collect_parameters
from cli.utils.session_manager import session_manager


@click.command()
def add_processing():
    """Add or update a processing configuration."""
    processing_name = input("Processing name: ")

    processing_data = {
        "processing_type": input("Processing type [default=DATASET_VERSION_CREATION]: ")
        or "DATASET_VERSION_CREATION",
        "picsellia_pipeline_script_path": input(
            "Picsellia pipeline script path [default=examples/processing/augmentation/augmentations_pipeline.py]: "
        )
        or "examples/processing/augmentation/augmentations_pipeline.py",
        "local_pipeline_script_path": input(
            "Local pipeline script path [default=examples/processing/augmentation/local_augmentations_pipeline.py]: "
        )
        or "examples/processing/augmentation/local_augmentations_pipeline.py",
        "requirements_path": input(
            "Requirements file path [default=examples/processing/augmentation/requirements.txt]: "
        )
        or "examples/processing/augmentation/requirements.txt",
        "image_name": input("Docker image name: "),
        "image_tag": input("Docker image tag [default=latest]: ") or "latest",
        "parameters": {},
    }

    # Collect parameters
    parameters_mode = (
        input("Enter 'manual', 'file', or 'none' for parameters [default=none]: ")
        or "none"
    )
    if parameters_mode == "manual":
        parameters: dict[str, str] = {}
        while True:
            key = input("Parameter key (leave empty to stop): ")
            if not key:
                break
            value = input(f"Value for '{key}': ")
            parameters[key] = value
        processing_data["parameters"] = parameters
    elif parameters_mode == "file":
        json_path = input("Path to JSON file: ")
        processing_data["parameters"] = collect_parameters("file", json_path)
    else:
        processing_data["parameters"] = {}

    # Save processing
    session_manager.add_processing(processing_name, processing_data)
    print(f"Processing '{processing_name}' added or updated successfully!")


@click.command()
@click.argument("name")
def remove_processing(name):
    """Delete a processing configuration."""
    try:
        session_manager.remove_processing(name)
        click.echo(f"Processing '{name}' deleted successfully!")
    except KeyError as e:
        click.echo(str(e))


@click.command()
def list_processings():
    """List all registered processings."""
    processings = session_manager.list_processings()
    if not processings:
        click.echo("No processings registered.")
    else:
        click.echo("Registered processings:")
        for processing in processings:
            click.echo(f"  - {processing}")
