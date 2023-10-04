import os
import time
from typing import TextIO, Generator, Any, Tuple


def is_string_valid(line: str) -> bool:
    """
    Check if a line is valid or not.

    A line is considered invalid if it's None, an empty string,
    a string composed only of backspaces, or a string composed
    only of a newline character.

    Args:
        line (str): The line to check.

    Returns:
        bool: True if the line is valid, False otherwise.
    """
    line = line.rstrip("\n")
    return not (not line or all(ord(char) == 8 for char in line))


def format_line(line: str) -> str:
    """
    Format the given line by removing trailing backspaces and ensuring it ends with a newline.

    Args:
        line (str): The line to be formatted. It's assumed that this is not None.

    Returns:
        str: The formatted line. If the input line ended with one or more backspaces, these are removed.
        If the input line did not end with a newline, one is added.
    """
    return line.rstrip("\n").rstrip(chr(8)) + "\n"


class LogTailer:
    SLEEP_INTERVAL = 0.04

    def __init__(self, log_file: TextIO):
        self.log_file = log_file
        self.progress_bar = []

    def _process_line(self, line: str) -> Tuple[str, bool, bool]:
        """
        Process a line from the log file.

        Args:
            line (str): The line to process.

        Returns:
            A tuple containing the processed line, a boolean indicating whether the line should replace the previous
        """
        replace_log = False
        is_first_line = False
        line = format_line(line)

        if "it" in line and "%" in line and "<" in line:
            replace_log = True
            if not self.progress_bar:
                is_first_line = True
            else:
                self.progress_bar[-1] += "\r"
            self.progress_bar.append(line)
            line = "".join(self.progress_bar)
            if "100%" in line:
                self.progress_bar = []

        return line, replace_log, is_first_line

    def tail(self) -> Generator[Tuple[str, bool, bool], Any, None]:
        """
        Stream the content of the provided file_path
        Args:

        Returns:
            A generator that yields newly added lines from the log_file as they are written.
        """
        self.log_file.seek(0, os.SEEK_END)

        while True:
            line = self.log_file.readline()
            if not is_string_valid(line):
                time.sleep(self.SLEEP_INTERVAL)
                continue

            yield self._process_line(line)
