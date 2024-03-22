import pytest

from src import Pipeline
from tests.decorators.fixtures.pipeline_fixtures import (
    reset_pipeline_entrypoint_call_tracker,
)

pytest_plugins = [
    "tests.decorators.fixtures.pipeline_fixtures",
    "tests.decorators.fixtures.step_fixtures",
]


@pytest.fixture(autouse=True)
def cleanup():
    Pipeline.ACTIVE_PIPELINE = None
    Pipeline.STEPS_REGISTRY.clear()
    reset_pipeline_entrypoint_call_tracker()
    yield
