import uuid

import pytest

from src.decorators.step_decorator import Step
from src.enums import StepState
from src.models.steps.step_metadata import StepMetadata


def step_entrypoint_pass():
    pass


def step_entrypoint_fail():
    raise Exception("Step Failed")


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
def step_continue_on_failure():
    metadata = StepMetadata(
        id=uuid.uuid4(),
        name="continue_on_failure_step",
        display_name="Continue On Failure Step",
        state=StepState.PENDING,
    )
    return Step(
        continue_on_failure=True,
        entrypoint=step_entrypoint_pass,
        metadata=metadata,
    )


@pytest.fixture
def step_no_continue_on_failure():
    metadata = StepMetadata(
        id=uuid.uuid4(),
        name="no_continue_on_failure_step",
        display_name="No Continue On Failure Step",
        state=StepState.PENDING,
    )
    return Step(
        continue_on_failure=False,
        entrypoint=step_entrypoint_fail,
        metadata=metadata,
    )
