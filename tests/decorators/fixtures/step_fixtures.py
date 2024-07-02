import random
import string
import uuid

import pytest

from src.decorators.step_decorator import Step
from src.enums import StepState
from src.models.steps.step_metadata import StepMetadata


def step_entrypoint_pass():
    pass


def step_entrypoint_fail():
    raise RuntimeError("Step Failed")


def random_string(length=10):
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


@pytest.fixture
def mock_step_metadata_uuid() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def mock_step_metadata_name_1() -> str:
    return "mocked_step_metadata_name_1"


@pytest.fixture
def mock_step_metadata_name_2() -> str:
    return "mocked_step_metadata_name_2"


@pytest.fixture
def mock_step_continue_on_failure_1() -> bool:
    return False


@pytest.fixture
def mock_step_continue_on_failure_2() -> bool:
    return True


@pytest.fixture
def mock_step_metadata_1(mock_step_metadata_name_1: str):
    return StepMetadata(
        id=uuid.uuid4(),
        name=mock_step_metadata_name_1,
        display_name=mock_step_metadata_name_1,
        state=StepState.PENDING,
    )


@pytest.fixture
def mock_step_metadata_2(mock_step_metadata_name_2: str):
    return StepMetadata(
        id=uuid.uuid4(),
        name=mock_step_metadata_name_2,
        display_name=mock_step_metadata_name_2,
        state=StepState.PENDING,
    )


@pytest.fixture
def mock_step_entrypoint_pass():
    return step_entrypoint_pass


@pytest.fixture
def mock_step_entrypoint_fail():
    return step_entrypoint_fail


@pytest.fixture
def step_factory():
    """Fixture to dynamically create a Step instance with customizable parameters."""

    def _factory(continue_on_failure: bool, entrypoint):
        """
        The actual factory function.

        Args:
            continue_on_failure (bool): Determines if the step should continue on failure.
            entrypoint (Callable): The entrypoint function for the step.

        Returns:
            Step: An instance of Step configured as per the arguments.
        """
        # Generate a random name and display name
        name = random_string()
        display_name = name.replace("_", " ").capitalize()

        # Use the temp_file for the log file path
        metadata = StepMetadata(
            id=uuid.uuid4(),
            name=name,
            display_name=display_name,
            state=StepState.PENDING,
        )

        return Step(
            continue_on_failure=continue_on_failure,
            entrypoint=entrypoint,
            metadata=metadata,
        )

    return _factory
