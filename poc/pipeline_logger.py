import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from typing import Union

from poc.log_handler import StreamToLogger
from poc.step_metadata import StepMetadata


class LoggerManager:
    def __init__(self, pipeline_name: str, log_folder_path: Union[str, None]):
        self.pipeline_name = pipeline_name
        self.log_folder_root_path = log_folder_path
        self.uses_temp_dir = False

        self.steps_logger: {str: logging.Logger} = {}

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.current_file_handler = None

    def configure(self, steps_metadata: [StepMetadata]) -> None:
        """
        Configures the pipeline logger.
        If the log folder path is not provided, a temporary directory is created.
        """

        if self.log_folder_root_path is None:
            self.log_folder_root_path = tempfile.mkdtemp()
            self.uses_temp_dir = True

        elif not os.path.isdir(self.log_folder_root_path):
            raise NotADirectoryError(
                f"{self.log_folder_root_path} is not a valid directory."
            )

        self._create_pipeline_log_folder()
        self._configure_steps_logger(steps_metadata)

    def clean(self) -> None:
        """
        Cleans the log folder.
        If the log folder is a temporary directory, it is removed.
        Otherwise, only the pipeline log folder is removed.
        """
        if self.uses_temp_dir:
            os.rmdir(self.log_folder_root_path)
        else:
            os.rmdir(self.log_folder_path)

    def get_step_logger(self, step_id: str) -> logging.Logger:
        return self.steps_logger[step_id]

    def _create_pipeline_log_folder(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.log_folder_path = os.path.join(
            self.log_folder_root_path, f"{self.pipeline_name}_{timestamp}"
        )

        os.makedirs(self.log_folder_path)

    def _configure_steps_logger(self, steps_metadata: [StepMetadata]) -> None:
        for index, step_metadata in enumerate(steps_metadata):
            step_metadata.index = index + 1

            log_file_name = self._sanitize_file_path(
                input_string=f"{step_metadata.index}-{step_metadata.name}-{step_metadata.id}.txt"
            )
            step_log_file_path = os.path.join(self.log_folder_path, log_file_name)
            step_metadata.log_file_path = step_log_file_path

            open(step_log_file_path, "w").close()

    def _sanitize_file_path(self, input_string: str, replacement: str = "_"):
        invalid_chars_pattern = r'[\\/*?:"<>|]'
        sanitized_string = re.sub(invalid_chars_pattern, replacement, input_string)
        return sanitized_string

    def prepare_step_logger(self, log_file_path: str) -> logging.Logger:
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        for logger in logging.root.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                handlers = logger.handlers[:]
                for handler in handlers:
                    if isinstance(handler, logging.FileHandler):
                        logger.removeHandler(handler)
                        handler.close()

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        new_file_handler = logging.FileHandler(log_file_path, "a")
        new_file_handler.setFormatter(logging.Formatter("%(message)s"))

        for logger in logging.root.manager.loggerDict.values():
            if isinstance(logger, logging.Logger):
                logger.addHandler(new_file_handler)

        logging.getLogger().addHandler(new_file_handler)

        sys.stdout = StreamToLogger(log_file_path, self.original_stdout)
        sys.stderr = StreamToLogger(log_file_path, self.original_stderr)

        return logging.getLogger("picsellia")
