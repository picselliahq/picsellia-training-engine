import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import TextIO, Generator, Any

from picsellia import Client, Experiment, Job, Project
from picsellia.types.enums import ExperimentStatus, JobRunStatus

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

logging.getLogger("experiment").setLevel(logging.INFO)


def get_picsellia_client() -> Client:
    """
    Get Picsellia's client, used to interact with the experiment.

    Returns:
        A Picsellia Client object, ready to be used if the provided environment variables are correct.

    Raises:
        RuntimeError: If api_token is not available as an environment variable.
    """
    if "api_token" not in os.environ:
        raise RuntimeError("You must set an api_token to run this image")

    api_token = os.environ["api_token"]
    host = os.environ.get("host", "https://app.picsellia.com")
    organization_id = os.environ.get("organization_id")

    return Client(api_token=api_token, host=host, organization_id=organization_id)


def get_picsellia_job(client: Client) -> Job:
    """
    Get the job associated with the provided client.

    Args:
        client: A Picsellia's client instance.
        job_id: The id of the job you want to retrieve.

    Returns:
        A Job instance if a job id is provided, else None.
    """
    if job_id := os.environ.get("job_id"):
        job = client.get_job_by_id(job_id)
        job.update_job_run_with_status(JobRunStatus.RUNNING)
        return job

    return None


def get_picsellia_project(client: Client) -> Project:
    """
    Retrieved, if possible, the desired project.

    Args:
        client: A Picsellia's client instance.

    Returns:
        A Project instance if a project token or a project name is provided, else raises a RuntimeError.

    Raises:
        RuntimeError: If neither project_token nor project_name are provided as environment variables, as one of them is necessary to retrieve the Project instance.
    """
    if project_token := os.environ.get("project_token"):
        return client.get_project_by_id(project_token)
    elif project_name := os.environ.get("project_name"):
        return client.get_project(project_name)
    else:
        raise RuntimeError(
            "Cannot retrieve the Project. Either project_token or project_name must be specified."
        )


def get_picsellia_experiment(client: Client) -> Experiment:
    """
    Retrieved, if possible, the desired experiment.

    Args:
        client: A Picsellia's client instance.

    Returns:
        An Experiment instance if experiment_name is provided, else raises a RuntimeError.

    Raises:
        RuntimeError: If experiment_name is not available as an environment variable.
    """

    if experiment_name := os.environ.get("experiment_name"):
        picsellia_project = get_picsellia_project(client)
        return picsellia_project.get_experiment(experiment_name)
    else:
        raise RuntimeError(
            "Cannot retrieve the Experiment, environment variable experiment_name must be specified."
        )


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
    result = line.rstrip("\n").rstrip(chr(8))
    result += "\n"

    return result


def tail_f(log_file: TextIO) -> Generator[str, Any, None]:
    """
    Stream the content of the provided file_path
    Args:
        log_file: A reference to the opened log file.

    Returns:
        A generator that yields newly added lines from the log_file as they are written.
    """
    log_file.seek(0, os.SEEK_END)

    while True:
        line = log_file.readline()

        if not is_string_valid(line):
            time.sleep(0.04)
            continue
        
        line = format_line(line)

        yield line


def start_log_monitoring(client: Client, log_file_path: str):
    """
    Retrieves continuously what's inside the provided log file at log_file_path. Send it to the Picsellia's experiment's telemetry tab.

    Args:
        client: A Picsellia's client instance.
        log_file_path: The path to the log file.

    """
    job = get_picsellia_job(client)
    experiment = get_picsellia_experiment(client)

    section_header = "--#--Set up training"
    replace_log = False
    start_buffer = False

    buffer = []
    buffer_length = 0

    try:
        experiment.send_logging(section_header, section_header)
    except Exception:
        pass

    logs = {section_header: {"datetime": str(datetime.now().isoformat()), "logs": {}}}

    with open(log_file_path, "r") as log_file:
        for line in tail_f(log_file):
            exit_match = re.search(r"--ec-- ([0-9]+)", line)

            if exit_match:
                exit_code = int(exit_match.group(1))
                end_log_monitoring(
                    job=job,
                    experiment=experiment,
                    logs=logs,
                    buffer=buffer,
                    section_header=section_header,
                    exit_code=exit_code,
                )
                break

            if line.startswith("--#--"):
                logs[line] = {
                    "datetime": str(datetime.now().isoformat()),
                    "logs": {},
                }

            if line.startswith("-----"):
                progress_line_nb = experiment.line_nb
                replace_log = True

            if line.startswith("--*--"):
                replace_log = False

            if re.match("--[0-9]--", line[:6]):
                start_buffer = True
                buffer_length = int(line[2])

            if re.match("---[0-9]---", line[:8]):
                start_buffer = False
                try:
                    experiment.send_logging(buffer, section_header, special="buffer")
                    experiment.line_nb += len(buffer) - 1
                except Exception:
                    pass
                buffer = []

            if start_buffer:
                buffer.append(line)
                logs[section_header]["logs"][
                    str(experiment.line_nb + len(buffer))
                ] = line
                if len(buffer) == buffer_length:
                    try:
                        experiment.send_logging(
                            buffer, section_header, special="buffer"
                        )
                        experiment.line_nb += buffer_length - 1
                    except Exception:
                        pass
                    buffer = []
            else:
                if not replace_log:
                    try:
                        experiment.send_logging(line, section_header)
                        logs[section_header]["logs"][str(experiment.line_nb)] = line
                    except Exception:
                        pass
                else:
                    try:
                        experiment.line_nb = progress_line_nb
                        experiment.send_logging(line, section_header)
                    except Exception:
                        pass


def end_log_monitoring(
    job: Job,
    experiment: Experiment,
    logs: dict,
    buffer: list,
    section_header: str,
    exit_code: int,
):
    """
    Wrap the log monitoring up by sending the final json log file and updating the job status if necessary.

    Args:
        job: The job instance, if provided.
        experiment: The experiment associated with the training.
        logs: The retrieved logs.
        buffer:
        section_header:
        exit_code:

    Returns:

    """
    with open("{}-logs.json".format(experiment.id), "w") as json_log_file:
        if buffer:
            for i, line in enumerate(buffer):
                logs[section_header]["logs"][str(experiment.line_nb + i)] = line
            experiment.send_logging(buffer, section_header, special="buffer")

        logs["exit_code"] = {
            "exit_code": exit_code,
            "datetime": str(datetime.now().isoformat()),
        }
        json.dump(logs, json_log_file)

    experiment.send_logging(str(exit_code), section_header, special="exit_code")
    experiment.store_logging_file("{}-logs.json".format(experiment.id))

    if exit_code == 0:
        experiment.update(status=ExperimentStatus.SUCCESS)
        if job:
            job.update_job_run_with_status(JobRunStatus.SUCCEEDED)
    else:
        experiment.update(status=ExperimentStatus.FAILED)
        if job:
            job.update_job_run_with_status(JobRunStatus.FAILED)


def start(log_file_path):
    picsellia_client = get_picsellia_client()
    start_log_monitoring(client=picsellia_client, log_file_path=log_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file_path",
        help="Path to the log file where the training logs are written",
    )
    args = parser.parse_args()

    start(args.log_file_path)
