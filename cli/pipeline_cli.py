import click

from cli.commands.build_and_push_processing import build_and_push_processing
from commands.docker_pipeline import setup_dockerized_pipeline
from commands.processing import (
    add_processing,
    remove_processing,
    list_processings,
)
from commands.push_processing_on_picsellia import (
    push_processing_on_picsellia,
)
from commands.session import initialize_session
from commands.test_processing import test_processing


@click.group()
def cli():
    """CLI for managing pipelines with session support."""
    pass


cli.add_command(initialize_session)
cli.add_command(add_processing)
cli.add_command(remove_processing)
cli.add_command(list_processings)
cli.add_command(test_processing)
cli.add_command(setup_dockerized_pipeline)
cli.add_command(build_and_push_processing)
cli.add_command(push_processing_on_picsellia)

if __name__ == "__main__":
    cli()
