import click
import json
import os
from typing import Dict
import re


def collect_parameters(parameters_mode: str, json_path: str) -> Dict:
    """
    Collect parameters either manually or from a JSON file.
    """
    if parameters_mode == "manual":
        parameters = {}
        while True:
            param_name = click.prompt("Parameter name", type=str)
            param_value = click.prompt("Default value", type=str)
            parameters[param_name] = param_value
            more = click.confirm("Add another parameter?", default=True)
            if not more:
                break
    elif parameters_mode == "file":
        if not json_path or not os.path.isfile(json_path):
            raise click.ClickException("Invalid or missing JSON file path.")
        with open(json_path, "r") as file:
            parameters = json.load(file)
    else:
        raise click.ClickException("Invalid mode selected. Choose 'manual' or 'file'.")

    return parameters


def update_processing_parameters(script_path, new_parameters):
    """
    Update the processing_parameters dictionary in a pipeline script.

    Args:
        script_path (str): Path to the pipeline script.
        new_parameters (dict): New parameters to replace the existing ones.

    Returns:
        None
    """
    # Convert the new parameters dictionary to a string in Python dict format
    parameters_string = ",\n        ".join(
        [f'"{key}": "{value}"' for key, value in new_parameters.items()]
    )
    parameters_block = f"processing_parameters={{\n        {parameters_string}\n    }}"

    # Regular expression to find the processing_parameters block
    pattern = re.compile(r"processing_parameters=\{.*?\}", re.DOTALL)

    # Read the current script
    with open(script_path, "r") as file:
        content = file.read()

    # Replace the processing_parameters block
    updated_content = pattern.sub(parameters_block, content)

    # Write the updated script back to the file
    with open(script_path, "w") as file:
        file.write(updated_content)

    print(f"Updated processing_parameters in '{script_path}' successfully!")
