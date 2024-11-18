import os
import time
from typing import Any, Generator, TextIO, Tuple


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


def is_progress_line(line: str) -> bool:
    symbols = ["%", "<", "[", "]"]
    return all(symbol in line for symbol in symbols)


class LogTailer:
    SLEEP_INTERVAL = 0.04

    def __init__(self, log_file: TextIO):
        self.log_file = log_file
        self.progress_bar = []

    def _process_line(
        self, line: str, wait_for_done: bool
    ) -> Tuple[str, bool, bool, bool]:
        replace_log = False
        is_first_line = False
        line = format_line(line)

        if is_progress_line(line):
            replace_log = True
            if "100%" not in line and wait_for_done:
                self.progress_bar = []
                wait_for_done = False
            if not self.progress_bar:
                is_first_line = True
            self.progress_bar = [line]

            if "100%" in line:
                wait_for_done = True

        return line, replace_log, is_first_line, wait_for_done

    def tail(self) -> Generator[Tuple[str, bool, bool], Any, None]:
        self.log_file.seek(0, os.SEEK_END)

        wait_for_done = False

        while True:
            line = self.log_file.readline()
            if not is_string_valid(line):
                time.sleep(self.SLEEP_INTERVAL)
                continue

            line, replace_log, is_first_line, wait_for_done = self._process_line(
                line, wait_for_done
            )

            yield line, replace_log, is_first_line
