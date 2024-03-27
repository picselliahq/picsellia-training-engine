import pytest

from src.logger import LoggerManager


@pytest.fixture
def logger_manager(temp_log_dir):
    return LoggerManager(
        pipeline_name="TestPipeline", log_folder_root_path=temp_log_dir
    )
