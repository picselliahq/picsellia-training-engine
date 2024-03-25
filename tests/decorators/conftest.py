import os
import shutil

from src import Pipeline
from tests.decorators.fixtures.pipeline_fixtures import (
    reset_pipeline_entrypoint_call_tracker,
)
import pytest


@pytest.fixture
def temp_log_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "logs")


@pytest.fixture(autouse=True)
def cleanup(temp_log_dir):
    Pipeline.ACTIVE_PIPELINE = None
    Pipeline.STEPS_REGISTRY.clear()
    reset_pipeline_entrypoint_call_tracker()
    os.makedirs(temp_log_dir, exist_ok=True)
    yield

    if os.path.isdir(temp_log_dir):
        shutil.rmtree(temp_log_dir)
