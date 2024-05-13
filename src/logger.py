import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from typing import Optional, TextIO, cast, List

from src.models.logging.stream_to_logger import StreamToLogger
from src.models.steps.step_metadata import StepMetadata


class LoggerManager:
    def __init__(self, pipeline_name: str, log_folder_root_path: Optional[str] = None):
        self.pipeline_name = pipeline_name
        self.log_folder_root_path = log_folder_root_path
        self.log_folder_path: Optional[str] = None
        self.uses_temp_dir = False

        self.original_stdout: TextIO = sys.stdout
        self.original_stderr: TextIO = sys.stderr
        self.logger = logging.getLogger("picsellia-engine")
        self.logger.setLevel(logging.INFO)

    def clean(self) -> None:
        """
        Cleans the log folder. If the log folder is a temporary directory, it is removed.
        Otherwise, only the pipeline log folder is removed.
        """
        if self.uses_temp_dir and self.log_folder_root_path:
            shutil.rmtree(self.log_folder_root_path)
        elif self.log_folder_path:
            shutil.rmtree(self.log_folder_path)
        else:
            self.logger.warning("No log folder could cleaned.")

    def configure_pipeline_initialization_log_file(self) -> str:
        """Configures the log file for the pipeline own logs.

        Returns:
            The path of the log file where the pipeline will log its own logs.
        """
        if self.log_folder_path is None:
            raise ValueError(
                "The log folder path is empty, did you forget to call `configure_log_files`?"
            )

        log_file_name = "0-pipeline-initialization.log"
        pipeline_initialization_log_file_path = os.path.join(
            self.log_folder_path, log_file_name
        )
        open(pipeline_initialization_log_file_path, "w").close()

        return pipeline_initialization_log_file_path

    def configure_log_files(self, steps_metadata: List[StepMetadata]) -> None:
        """Configures the folders and log files for the pipeline and its steps.

        If the `log_folder_path` is not provided when decorating a pipeline, a temporary directory is created instead.
        This function will create the log folder and configure the log files for each step.

        Args:
            steps_metadata: The metadata of the steps in the pipeline.
        """

        if self.log_folder_root_path is None:
            self.log_folder_root_path = tempfile.mkdtemp()
            self.uses_temp_dir = True

        elif not os.path.isdir(self.log_folder_root_path):
            os.makedirs(self.log_folder_root_path)
            self.logger.info(
                f"Log folder created at {os.path.abspath(self.log_folder_root_path) }"
            )

        self._create_pipeline_log_folder()
        self._configure_steps_log_files(steps_metadata=steps_metadata)

    def prepare_logger(self, log_file_path: Optional[str]) -> logging.Logger:
        """
        Prepares the logger for a step or the pipeline.

        The logging strategy is to log to the console (`StreamToLogger`) and to a file (`FileHandler`) at the same time.
        Each time a new step begins, the loggers are reset to remove the previous file handler and add the new one.

        Args:
            log_file_path: The path of the log file where the logs will be written.

        Returns:
            The configured logger.
        """
        if not log_file_path:
            raise ValueError(
                "The log file path is empty. Did you forget to call `configure_log_files`?"
            )

        if not os.path.isfile(log_file_path):
            raise FileNotFoundError(
                f"Cannot open log file at {log_file_path}. This file does not exist."
            )

        # Reset the stdout and stderr to the original streams, this is needed to avoid issues with the logger
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # Remove the file handlers from the previous step and close them
        for logger in logging.root.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                self._reset_file_handlers(logger=logger)

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Prepare the file handler for the next step
        new_file_handler = logging.FileHandler(log_file_path, "a")
        new_file_handler.setFormatter(fmt=logging.Formatter("%(message)s"))

        # Add the file handler to all the loggers
        for logger in logging.root.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.addHandler(hdlr=new_file_handler)

        # Prepare as well the root logger
        root_logger = logging.getLogger()
        self._reset_file_handlers(logger=root_logger)
        root_logger.addHandler(hdlr=new_file_handler)

        # Redirect stdout and stderr to the logger
        sys.stdout = cast(
            TextIO,
            StreamToLogger(
                filepath=log_file_path, original_stream=self.original_stdout
            ),
        )
        sys.stderr = cast(
            TextIO,
            StreamToLogger(
                filepath=log_file_path, original_stream=self.original_stderr
            ),
        )

        return self.logger

    def _configure_steps_log_files(self, steps_metadata: List[StepMetadata]) -> None:
        """Configures the log files for each step.

        For each step, will look at the metadata to create a log file.
        The log file's name is composed of the step index, name and id. Example: `1-step_name-uuid.log`.

        Args:
            steps_metadata: The metadata of the steps in the pipeline.
        """
        if self.log_folder_path is None:
            raise ValueError(
                "The log folder path is empty, did you forget to call `_create_pipeline_log_folder`?"
            )
        for index, step_metadata in enumerate(steps_metadata):
            step_metadata.index = index + 1

            log_file_name = self._sanitize_file_path(
                input_string=f"{step_metadata.index}-{step_metadata.name}-{step_metadata.id}.log"
            )
            step_log_file_path = os.path.join(self.log_folder_path, log_file_name)
            step_metadata.log_file_path = step_log_file_path

            open(step_log_file_path, "w").close()

    def _create_pipeline_log_folder(self) -> None:
        """Creates the log folder for the pipeline. The folder name is composed of the pipeline name and a timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

        if self.log_folder_root_path is None:
            raise ValueError(
                "The log folder root path is empty, did you forget to call `configure_log_files`?"
            )

        self.log_folder_path = os.path.join(
            self.log_folder_root_path, f"{self.pipeline_name}_{timestamp}"
        )

        os.makedirs(self.log_folder_path)

    def _reset_file_handlers(self, logger: logging.Logger):
        """Resets the file handlers of the logger.

        This function will remove all the file handlers from the provided logger and close them.

        Args:
            logger: The logger to reset the file handlers of.
        """
        handlers = logger.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()

    def _sanitize_file_path(self, input_string: str, replacement: str = "_") -> str:
        """Sanitizes a string to be used as a file path.

        Args:
            input_string: The string to sanitize.
            replacement: The character to replace invalid characters with.

        Returns:
            The sanitized string.
        """
        invalid_chars_pattern = r'[\\/*?:"<>|]'
        sanitized_string = re.sub(invalid_chars_pattern, replacement, input_string)
        return sanitized_string
