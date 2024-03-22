from typing import Optional, Callable

import pytest

from src import Pipeline

entrypoint_call_tracker = {
    "count": 0,
    "args": [],
    "kwargs": {},
}


def pipeline_entrypoint(*args, **kwargs) -> None:
    """Mock pipeline entrypoint function that tracks calls and arguments."""
    entrypoint_call_tracker["count"] += 1
    if args:
        entrypoint_call_tracker["args"].append(*args)

    if kwargs:
        entrypoint_call_tracker["kwargs"].update(**kwargs)


# Ensure to reset the tracker as necessary, especially if using with multiple tests
def reset_pipeline_entrypoint_call_tracker():
    entrypoint_call_tracker["count"] = 0
    entrypoint_call_tracker["args"].clear()
    entrypoint_call_tracker["kwargs"].clear()


@pytest.fixture
def mock_pipeline_entrypoint_source() -> str:
    return """
def mock_entrypoint():
    step1()
    step2()
    """


@pytest.fixture
def mock_pipeline_context() -> dict:
    return {}


@pytest.fixture
def mock_pipeline_name() -> str:
    return "TestPipeline"


@pytest.fixture
def mock_pipeline_log_folder_path() -> Optional[str]:
    return None


@pytest.fixture
def mock_pipeline_remove_logs_on_completion() -> bool:
    return True


@pytest.fixture
def mock_pipeline_entrypoint():
    return pipeline_entrypoint


@pytest.fixture
def mock_pipeline(
    mock_pipeline_context: dict,
    mock_pipeline_name: str,
    mock_pipeline_log_folder_path,
    mock_pipeline_remove_logs_on_completion: bool,
    mock_pipeline_entrypoint: Callable,
):
    return Pipeline(
        context=mock_pipeline_context,
        name=mock_pipeline_name,
        log_folder_path=mock_pipeline_log_folder_path,
        remove_logs_on_completion=mock_pipeline_remove_logs_on_completion,
        entrypoint=mock_pipeline_entrypoint,
    )
