import os

from picsellia import Client, Job
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


def get_picsellia_job(client: Client) -> Job | None:
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
