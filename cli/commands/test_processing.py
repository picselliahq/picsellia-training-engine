import os
import subprocess

from cli.helpers.prompt import fetch_processing_name
from cli.helpers.session import ensure_session_initialized
from cli.helpers.validation import validate_and_update_processing
from cli.utils.collect_params import update_processing_parameters
from cli.utils.session_manager import session_manager
import click

from cli.utils.dockerfile_generation import get_repository_root


@click.command()
def test_processing():
    """
    Test the local processing pipeline script with specified arguments.
    """
    ensure_session_initialized()

    processing_name = fetch_processing_name()
    if not processing_name:
        return

    processing = validate_and_update_processing(processing_name)
    if not processing:
        return

    results_dir = click.prompt("Results directory", type=click.Path())
    if os.path.exists(results_dir):
        override = click.confirm(
            f"Directory {results_dir} already exists. Do you want to override it?"
        )
        if not override:
            results_dir = click.prompt("Results directory", type=click.Path())
        else:
            os.system(f"rm -rf {results_dir}")
            os.makedirs(results_dir, exist_ok=True)
    input_dataset_version_id = click.prompt("Input dataset version ID", type=str)
    output_dataset_version_name = click.prompt(
        "Output dataset version name that will be created", type=str
    )

    global_data = session_manager.get_global()

    try:
        repo_root = get_repository_root()

        pipeline_script = os.path.join(
            repo_root, processing["local_pipeline_script_path"]
        )
        if not os.path.exists(pipeline_script):
            raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

        update_processing_parameters(
            os.path.join(repo_root, processing["local_pipeline_script_path"]),
            processing["parameters"],
        )

        command = [
            "python",
            str(pipeline_script),
            "--api_token",
            global_data["api_token"],
            "--organization_id",
            global_data["organization_id"],
            "--results_dir",
            results_dir,
            "--job_type",
            processing["processing_type"],
            "--input_dataset_version_id",
            input_dataset_version_id,
            "--output_dataset_version_name",
            output_dataset_version_name,
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)

        click.echo(f"Running the local pipeline script with PYTHONPATH={repo_root}...")
        subprocess.run(command, check=True, env=env)
        click.echo(f"Processing '{processing_name}' tested successfully!")
    except FileNotFoundError as e:
        click.echo(f"File error: {e}")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error running the pipeline script: {e}")
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
