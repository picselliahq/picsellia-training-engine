import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Type

import picsellia
from picsellia import Experiment

from poc.models.parameters.augmentation_parameters import AugmentationParameters
from poc.models.parameters.hyper_parameters import HyperParameters


class PicselliaContext(ABC):
    def __init__(
        self, api_token=None, host=None, organization_id=None, organization_name=None
    ):
        self.api_token = api_token or os.getenv("api_token")

        if not self.api_token:
            raise ValueError(
                "API token not provided. Please provide it as an argument or set the 'api_token' environment variable."
            )

        self.host = host or os.getenv("host", "https://app.picsellia.com")
        self.organization_id = organization_id or os.getenv("organization_id")
        self.organization_name = organization_name or os.getenv("organization_name")

        self.client = self._initialize_client()

    def _initialize_client(self):
        return picsellia.Client(
            api_token=self.api_token,
            host=self.host,
            organization_id=self.organization_id,
            organization_name=self.organization_name,
        )

    @abstractmethod
    def to_dict(self):
        pass


class PicselliaTrainingContext(PicselliaContext):
    def __init__(
        self,
        hyperparameters_cls: Type[HyperParameters],
        augmentation_parameters_cls: Type[AugmentationParameters],
        api_token=None,
        host=None,
        organization_id=None,
        experiment_id=None,
    ):
        super().__init__(api_token, host, organization_id)
        self.experiment_id = experiment_id or os.getenv("experiment_id")

        if not self.experiment_id:
            raise ValueError(
                "Experiment ID not provided."
                "Please provide it as an argument or set the 'experiment_id' environment variable."
            )

        self.experiment = self._initialize_experiment()
        parameters_log_data = self.experiment.get_log("parameters").data

        self.hyperparameters = hyperparameters_cls(log_data=parameters_log_data)
        self.augmentation_parameters = augmentation_parameters_cls(
            log_data=parameters_log_data
        )

    def _initialize_experiment(self) -> Experiment:
        return self.client.get_experiment_by_id(self.experiment_id)

    def to_dict(self):
        result = defaultdict(dict)
        result["context_parameters"] = {
            "host": self.host,
            "organization_id": self.organization_id,
            "organization_name": self.organization_name,
            "experiment_id": self.experiment_id,
        }

        for key, value in self.hyperparameters.to_dict().items():
            suffix = " (default)" if key in self.hyperparameters.defaulted_keys else ""
            color_code = "\033[33m" if suffix else "\033[34m"
            result["hyperparameters"][key] = f"{color_code}{value}\033[0m{suffix}"

            # Augmentation Parameters
        for key, value in self.augmentation_parameters.to_dict().items():
            suffix = (
                " (default)"
                if key in self.augmentation_parameters.defaulted_keys
                else ""
            )
            color_code = "\033[33m" if suffix else "\033[34m"
            result["augmentation_parameters"][
                key
            ] = f"{color_code}{value}\033[0m{suffix}"

        return result


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

    def to_dict(self):
        return {
            "host": self.host,
            "organization_id": self.organization_id,
            "job_id": self.job_id,
        }
