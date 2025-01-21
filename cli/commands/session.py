import click

from cli.utils.session_manager import session_manager


@click.command()
def initialize_session():
    """Initialize global session data."""
    session_manager.ensure_session_initialized()
    print("Global session initialized successfully!")
