import os
import tempfile
from unittest.mock import patch
import pytest
from src.logger import LoggerManager


class TestLoggerManager:
    def test_configure_uses_provided_log_dir(self, logger_manager, temp_log_dir):
        logger_manager.configure_log_files(steps_metadata=[])
        assert not logger_manager.uses_temp_dir
        assert logger_manager.log_folder_root_path == temp_log_dir

    def test_configure_creates_provided_log_dir_if_not_exist(
        self, logger_manager, temp_log_dir
    ):
        logger_manager.log_folder_root_path = os.path.join(temp_log_dir, "nonexistent")
        logger_manager.configure_log_files(steps_metadata=[])
        assert os.path.exists(logger_manager.log_folder_root_path)

    def test_configure_uses_temp_log_dir(self, logger_manager):
        with patch("tempfile.mkdtemp", return_value="/tmp/testdir") as mkdtemp_mock:
            logger_manager.log_folder_root_path = None
            logger_manager.configure_log_files(steps_metadata=[])
            assert logger_manager.uses_temp_dir
            assert logger_manager.log_folder_root_path == mkdtemp_mock.return_value

    def test_clean_removes_temp_dir(self, logger_manager):
        logger_manager.uses_temp_dir = True
        logger_manager.log_folder_root_path = tempfile.mkdtemp()
        with patch("shutil.rmtree") as rmtree_mock:
            logger_manager.clean()
            rmtree_mock.assert_called_with(logger_manager.log_folder_root_path)

    def test_configure_pipeline_initialization_log_file_creates_file(
        self, logger_manager
    ):
        logger_manager.log_folder_path = tempfile.mkdtemp()
        log_file_path = logger_manager.configure_pipeline_initialization_log_file()
        assert os.path.exists(log_file_path)
        assert log_file_path.endswith("0-pipeline-initialization.log")

    def test_configure_pipeline_initialization_log_file_raises_error_if_no_log_folder_path(
        self, logger_manager
    ):
        with pytest.raises(ValueError):
            logger_manager.configure_pipeline_initialization_log_file()

    def test_configure_steps_logs_files_raises_error_if_no_log_folder_path(
        self, logger_manager
    ):
        with pytest.raises(ValueError):
            logger_manager._configure_steps_log_files(steps_metadata=[])

    def test_prepare_logger_raises_error_if_no_log_file_path(self, logger_manager):
        with pytest.raises(ValueError):
            logger_manager.prepare_logger(log_file_path=None)

    def test_prepare_logger_raises_error_if_log_file_path_does_not_exist(
        self, logger_manager
    ):
        with pytest.raises(FileNotFoundError):
            logger_manager.prepare_logger(log_file_path="nonexistent.log")

    def test_create_pipeline_log_folder_raises_error_if_no_log_folder_root_path(
        self, logger_manager
    ):
        logger_manager.log_folder_root_path = None

        with pytest.raises(ValueError):
            logger_manager._create_pipeline_log_folder()

    def test_configure_steps_log_files_creates_files(
        self, logger_manager, mock_step_metadata_1, mock_step_metadata_2, temp_log_dir
    ):
        logger_manager.log_folder_path = temp_log_dir
        steps_metadata = [mock_step_metadata_1, mock_step_metadata_2]
        logger_manager._configure_steps_log_files(steps_metadata)

        for step_metadata in steps_metadata:
            assert os.path.exists(step_metadata.log_file_path)

    @pytest.mark.parametrize(
        "input_string,expected_output",
        [
            ("normal_filename.log", "normal_filename.log"),
            ("filename_with_invalid:chars?.log", "filename_with_invalid_chars_.log"),
            (
                "path/with/slash\\and\\backslash.log",
                "path_with_slash_and_backslash.log",
            ),
            ('<invalid>|chars*in"filename.log', "_invalid__chars_in_filename.log"),
            ("nochange-needed.log", "nochange-needed.log"),
            ('multiple<>:"/\\|?*characters', "multiple_________characters"),
            ("ends_with_invalid_char?", "ends_with_invalid_char_"),
            ("|starts_with_invalid_char", "_starts_with_invalid_char"),
        ],
    )
    def test_sanitize_file_path(self, input_string, expected_output):
        logger_manager = LoggerManager(
            pipeline_name="TestPipeline", log_folder_root_path="/some/path"
        )

        sanitized_string = logger_manager._sanitize_file_path(input_string)

        assert (
            sanitized_string == expected_output
        ), f"Sanitization incorrect. Expected: '{expected_output}', got: '{sanitized_string}'"
