from cli.utils.session_manager import session_manager

import click


def fetch_processing_name():
    """
    Prompt the user to select a processing name from registered processings.
    Returns:
        str: The selected processing name.
    """
    registered_processings = session_manager.list_processings()
    if not registered_processings:
        click.echo(
            "No processings are registered. Please add one using the `add-processing` command."
        )
        return None
    return click.prompt(
        f"Processing name (choose from: [{', '.join(registered_processings)}])"
    )
