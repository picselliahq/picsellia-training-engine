from cli.utils.session_manager import session_manager

import click


def validate_and_update_processing(processing_name):
    """
    Validate the processing details and prompt the user to update them if necessary.
    Args:
        processing_name (str): The name of the processing to validate.
    Returns:
        dict: The validated (and possibly updated) processing details.
    """
    processing = session_manager.get_processing(processing_name)
    if not processing:
        click.echo(f"Processing '{processing_name}' not found.")
        return None

    # Display current details
    click.echo(f"Current details for processing '{processing_name}':")
    for key, value in processing.items():
        click.echo(f"  {key}: {value}")

    # Ask if the user wants to update
    confirm_update = click.prompt(
        "Do you want to update these details?",
        type=click.Choice(["yes", "no"], case_sensitive=False),
        default="no",
    )

    if confirm_update == "yes":
        click.echo(
            "\nUpdating processing details. Leave blank to keep the current value."
        )
        updated_processing = {}

        for key, value in processing.items():
            # Special handling for parameters
            if key == "parameters":
                updated_processing[key] = update_parameters(value)
                continue

            new_value = input(f"{key} [current: {value}]: ")
            updated_processing[key] = new_value if new_value else value

        # Save updated details
        session_manager.add_processing(processing_name, updated_processing)
        click.echo(f"Processing '{processing_name}' updated successfully!")

    return processing


def update_parameters(current_parameters):
    """
    Handle updates, additions, and deletions for parameters.
    Args:
        current_parameters (dict): The current parameters.
    Returns:
        dict: The updated parameters.
    """
    click.echo("\nCurrent parameters:")
    for key, value in current_parameters.items():
        click.echo(f"  {key}: {value}")

    updated_parameters = current_parameters.copy()

    # Loop through each existing parameter and ask if the user wants to update or delete it
    for key in list(current_parameters.keys()):
        action = click.prompt(
            f"Do you want to update, delete, or keep '{key}'?",
            type=click.Choice(["update", "delete", "keep"], case_sensitive=False),
            default="keep",
        )

        if action == "update":
            new_value = input(
                f"New value for '{key}' [current: {current_parameters[key]}]: "
            )
            updated_parameters[key] = (
                new_value if new_value else current_parameters[key]
            )
        elif action == "delete":
            del updated_parameters[key]

    # Allow the user to add new parameters
    click.echo("\nAdd new parameters (leave key blank to stop):")
    while True:
        new_key = input("New parameter key: ")
        if not new_key:
            break
        new_value = input(f"Value for '{new_key}': ")
        updated_parameters[new_key] = new_value

    click.echo("\nUpdated parameters:")
    for key, value in updated_parameters.items():
        click.echo(f"  {key}: {value}")

    return updated_parameters
