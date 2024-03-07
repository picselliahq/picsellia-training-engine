import os
import tempfile
from datetime import datetime
from typing import Union


class PipelineLogger:
    def __init__(self, pipeline_name: str, log_folder_path: Union[str, None]):
        self.pipeline_name = pipeline_name
        self.log_folder_root_path = log_folder_path
        self.uses_temp_dir = False

    def configure(self) -> None:
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

    def _create_pipeline_log_folder(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.log_folder_path = os.path.join(
            self.log_folder_root_path, f"{self.pipeline_name}_{timestamp}"
        )

        os.makedirs(self.log_folder_path)
