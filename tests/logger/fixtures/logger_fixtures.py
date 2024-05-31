import os
import shutil

import pytest

from src.logger import LoggerManager


@pytest.fixture
def logger_manager(temp_log_dir):
    return LoggerManager(
        pipeline_name="TestPipeline", log_folder_root_path=temp_log_dir
    )


@pytest.fixture(autouse=True)
def cleanup(temp_log_dir):
    os.makedirs(temp_log_dir, exist_ok=True)
    yield

    if os.path.isdir(temp_log_dir):
        shutil.rmtree(temp_log_dir)
