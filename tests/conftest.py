import logging
import pathlib

import pytest


def discover_fixture_modules(root_dir: str) -> list:
    """
    Discovers and formats fixture module paths for dynamic import.

    Args:
        root_dir (str): The root directory to start searching from.

    Returns:
        list: A list of dot-formatted module strings.
    """
    root_path = pathlib.Path(".")
    fixture_files = root_path.glob("**/fixtures/*_fixtures.py")
    module_strings = []

    for fixture_file in fixture_files:
        # Convert file path to module path relative to root_dir
        relative_path = fixture_file.relative_to(root_path)
        # Convert to dot notation and strip the '.py' extension
        module_str = str(relative_path).replace("/", ".").replace("\\", ".")[:-3]

        if not module_str.startswith(root_dir):
            module_str = f"{root_dir}.{module_str}"

        module_strings.append(module_str)

    return module_strings


pytest_plugins = discover_fixture_modules("tests")


@pytest.fixture
def test_logger() -> logging.Logger:
    return logging.getLogger("test_logger")
