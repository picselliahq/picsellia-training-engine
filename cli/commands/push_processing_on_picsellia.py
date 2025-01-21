import click

from cli.utils.prompt import fetch_processing_name
from cli.utils.validation import validate_and_update_processing
from cli.utils.session_manager import session_manager

from picsellia import Client
from picsellia.types.enums import ProcessingType


@click.command()
def push_processing_on_picsellia():
    """
    Push a processing pipeline to Picsellia.
    """
    session_manager.ensure_session_initialized()

    processing_name = fetch_processing_name()
    if not processing_name:
        return

    processing = validate_and_update_processing(processing_name)
    if not processing:
        return

    default_cpu = click.prompt("Default CPU", default=4, type=int, show_default=True)
    default_gpu = click.prompt("Default GPU", default=0, type=int, show_default=True)

    global_data = session_manager.get_global()

    try:
        client = Client(
            api_token=global_data["api_token"],
            organization_id=global_data["organization_id"],
        )

        client.create_processing(
            name=processing_name,
            type=ProcessingType(processing["processing_type"]),
            default_cpu=default_cpu,
            default_gpu=default_gpu,
            default_parameters=processing["parameters"],
            docker_image=processing["image_name"],
            docker_tag=processing["image_tag"],
            docker_flags=None,
        )
        click.echo(f"Processing '{processing_name}' pushed successfully!")
    except Exception as e:
        click.echo(f"Error pushing processing to Picsellia: {e}")
