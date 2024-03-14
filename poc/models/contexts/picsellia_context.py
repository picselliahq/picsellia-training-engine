import os
from abc import ABC

import picsellia


class PicselliaContext(ABC):
    def __init__(self, api_token=None, host=None, organization_id=None):
        self.api_token = api_token or os.getenv("api_token")

        if not self.api_token:
            raise ValueError(
                "API token not provided. Please provide it as an argument or set the 'api_token' environment variable."
            )

        self.host = host or os.getenv("host", "https://app.picsellia.com")
        self.organization_id = organization_id or os.getenv("organization_id")

        self.client = self._initialize_client()

    def _initialize_client(self):
        return picsellia.Client(
            api_token=self.api_token,
            host=self.host,
            organization_id=self.organization_id,
        )


class PicselliaTrainingContext(PicselliaContext):
    def __init__(
        self, api_token=None, host=None, organization_id=None, experiment_id=None
    ):
        super().__init__(api_token, host, organization_id)
        self.experiment_id = experiment_id or os.getenv("experiment_id")

        if not self.experiment_id:
            raise ValueError(
                "Experiment ID not provided."
                "Please provide it as an argument or set the 'experiment_id' environment variable."
            )

        self.experiment = self._initialize_experiment()

    def _initialize_experiment(self):
        return self.client.get_experiment_by_id(self.experiment_id)


class PicselliaProcessingContext(PicselliaContext):
    def __init__(self, api_token=None, host=None, organization_id=None, job_id=None):
        # Initialize the Picsellia client from the base class
        super().__init__(api_token, host, organization_id)

        # If job_id is not passed as a parameter, try to get it from the environment variables
        self.job_id = job_id or os.environ.get("job_id")
        if not self.job_id:
            raise ValueError(
                "Job ID not provided. Please provide it as an argument or set the 'job_id' environment variable."
            )

        # Initialize job and context specifics
        self.job = self._initialize_job()
        self.context = self._initialize_context()

    def _initialize_job(self):
        # Use the client from the base class to fetch the job by ID
        return self.client.get_job_by_id(self.job_id)

    def _initialize_context(self):
        # Retrieve and return the context from the job
        context = self.job.sync()["dataset_version_processing_job"]

        self.model_version_id = context.get("model_version_id")
        self.dataset_version_id = context.get("input_dataset_version_id")

        if not self.model_version_id or not self.dataset_version_id:
            raise ValueError(
                "Failed to retrieve necessary context from the job. Please ensure the job is correctly configured."
            )
        return context
