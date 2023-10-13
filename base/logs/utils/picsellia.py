import os

from picsellia import Client, Job, Project, Experiment
from picsellia.types.enums import JobRunStatus


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


def get_picsellia_experiment(client: Client) -> Experiment | None:
    """
    Get the experiment associated with the provided client.

    Args:
        client: A Picsellia's client instance.

    Returns:
        An Experiment instance if an experiment id is provided, else None.
    """
    if experiment_name := os.environ.get("experiment_name"):
        picsellia_project = get_picsellia_project(client)
        return picsellia_project.get_experiment(experiment_name)
    elif experiment_id := os.environ.get("experiment_id"):
        return client.get_experiment_by_id(experiment_id)

    return None


def get_picsellia_job(client: Client) -> Job | None:
    """
    Get the job associated with the provided client.

    Args:
        client: A Picsellia's client instance.

    Returns:
        A Job instance if a job id is provided, else None.
    """
    if job_id := os.environ.get("job_id"):
        job = client.get_job_by_id(job_id)
        job.update_job_run_with_status(JobRunStatus.RUNNING)
        return job

    return None
