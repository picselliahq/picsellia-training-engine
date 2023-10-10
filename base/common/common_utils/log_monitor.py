import json
import re
from datetime import datetime
from typing import Match, Optional, List

from picsellia.types.enums import JobRunStatus, ExperimentStatus

from .log_tailer import LogTailer
from .picsellia_utils import (
    get_picsellia_job,
    get_picsellia_client,
    get_picsellia_experiment,
)


def starts_buffer(line: str) -> Match[str] | None:
    """Check if line starts a buffer."""
    return re.match("--[0-9]--", line[:6])


def ends_buffer(line: str) -> Match[str] | None:
    """Check if line ends a buffer."""
    return re.match("---[0-9]---", line[:8])


class LogMonitor:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

        self.client = get_picsellia_client()
        self.experiment = get_picsellia_experiment(self.client)
        self.job = get_picsellia_job(self.client)

        self.logs = {}

        self.buffer = []
        self.start_buffer = False
        self.buffer_length = 0

        self.progress_line_nb = 0

    def handle_exit_code(self, line: str) -> None:
        """Handles the exit code in the line."""
        exit_match = re.search(r"--ec-- ([0-9]+)", line)
        if exit_match:
            self.end_log_monitoring(int(exit_match.group(1)), "--#--End job")

    def process_line(
            self, line: str, section_header: str, replace_log: bool, is_first_line: bool
    ):
        """Process a line from the log file."""
        self.handle_exit_code(line)

        if replace_log and is_first_line:
            self.progress_line_nb = self.job.line_nb

        replace_log = self.process_line_prefixes(line, replace_log)
        self.process_buffers(line, section_header)

        if self.start_buffer:
            self.append_to_buffer(line, section_header)
        else:
            self.handle_log(line, section_header, replace_log)

    def process_line_prefixes(self, line: str, replace_log: bool):
        """Process line prefixes and updates logs."""
        if line.startswith("--#--"):
            self.initialize_log_section(line)
        elif line.startswith("-----"):
            self.progress_line_nb = self.job.line_nb
            replace_log = True
            line += "\r"
        elif line.startswith("--*--"):
            replace_log = False
        return replace_log

    def initialize_log_section(self, line: str):
        """Initialize a new log section."""
        self.logs[line] = {"datetime": str(datetime.now().isoformat()), "logs": {}}

    def process_buffers(self, line: str, section_header: str):
        """Process line buffers."""
        if starts_buffer(line):
            self.start_buffer = True
            self.buffer_length = int(line[2])
        elif ends_buffer(line):
            self.send_and_reset_buffer(section_header)

    def send_and_reset_buffer(self, section_header: str):
        """Send the current buffer and reset it."""
        self.send_logging(self.buffer, section_header, special="buffer")
        self.job.line_nb += len(self.buffer) - 1
        self.buffer = []
        self.start_buffer = False

    def append_to_buffer(self, line: str, section_header: str):
        """Append a line to the buffer and send it if the buffer is full."""
        self.buffer.append(line)
        self.logs[section_header]["logs"][
            str(self.job.line_nb + len(self.buffer))
        ] = line
        if len(self.buffer) == self.buffer_length:
            self.send_and_reset_buffer(section_header)

    def handle_log(self, line: str, section_header: str, replace_log: bool):
        """Handle the replacement log."""
        if replace_log:
            self.job.line_nb = self.progress_line_nb
        self.send_logging(line, section_header)

    def send_logging(
            self, content: str | List, section_header: str, special: Optional[str] = False
    ) -> None:
        """Send logging to job and update logs."""
        try:
            self.job.send_logging(content, section_header, special=special)
            self.logs[section_header]["logs"][str(self.job.line_nb)] = content
        except Exception as e:
            pass

    def end_log_monitoring(self, exit_code: int, section_header: str):
        """
        Wrap the log monitoring up by sending the final json log file and updating the job status if necessary.

        Args:
            exit_code (int): The exit code of the job.
            section_header (str): The section header of the log.

        Returns:
            None
        """
        with open("{}-logs.json".format(self.job.id), "w") as json_log_file:
            if self.buffer:
                for i, line in enumerate(self.buffer):
                    self.logs[section_header]["logs"][str(self.job.line_nb + i)] = line
                self.job.send_logging(self.buffer, section_header, special="buffer")

            self.logs["exit_code"] = {
                "exit_code": exit_code,
                "datetime": str(datetime.now().isoformat()),
            }
            json.dump(self.logs, json_log_file)

        self.job.store_logging_file("{}-logs.json".format(self.job.id))

        if exit_code == 0:
            self.job.update_job_run_with_status(JobRunStatus.SUCCEEDED)
            if self.experiment:
                self.experiment.update(status=ExperimentStatus.SUCCESS)
        else:
            self.job.update_job_run_with_status(JobRunStatus.FAILED)
            if self.experiment:
                self.experiment.update(status=ExperimentStatus.FAILED)

    def start_monitoring(self):
        if self.experiment:
            section_header = "--#--Set up training"
        else:
            section_header = "--#--Start job"
        self.job.send_logging(section_header, section_header)
        self.logs = {
            section_header: {"datetime": str(datetime.now().isoformat()), "logs": {}}
        }

        with open(self.log_file_path, "r") as log_file:
            log_tailer = LogTailer(log_file)
            for line, replace_log, is_first_line in log_tailer.tail():
                self.process_line(line, section_header, replace_log, is_first_line)
