import logging
import os

import pytest


@pytest.fixture
def temp_log_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "tests_logs")


@pytest.fixture
def test_logger() -> logging.Logger:
    return logging.getLogger("test_logger")
