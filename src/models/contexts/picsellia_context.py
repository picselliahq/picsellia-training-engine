import os
from abc import ABC, abstractmethod
from typing import Type, Optional, Any, Dict

import picsellia  # type: ignore
from picsellia import Experiment  # type: ignore

from src.models.parameters.augmentation_parameters import AugmentationParameters
from src.models.parameters.hyper_parameters import HyperParameters


class PicselliaContext(ABC):
    def __init__(
        self,
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        organization_id: Optional[str] = None,
        organization_name: Optional[str] = None,
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

    def _initialize_client(self) -> picsellia.Client:
        return picsellia.Client(
            api_token=self.api_token,
            host=self.host,
            organization_id=self.organization_id,
            organization_name=self.organization_name,
        )

    def _format_parameter_with_color_and_suffix(
        self, value: Any, key: str, defaulted_keys: set
    ) -> str:
        """
        Formats a given value with ANSI color codes and a suffix indicating whether it is a default value.

        Args:
            value: The value to be formatted. This could be of any type but is converted to a string.
            key: The key associated with the value. Used to determine if the value is defaulted.
            defaulted_keys: A set of keys that are considered to have default values.

        Returns:
            The value formatted as a string with appropriate color coding and suffix.
        """
        suffix = " (default)" if key in defaulted_keys else ""
        color_code = "\033[33m" if suffix else "\033[34m"
        return f"{color_code}{value}\033[0m{suffix}"

    def _process_parameters(self, parameters_dict: dict, defaulted_keys: set) -> dict:
        """
        Processes parameters by applying color coding and suffixes to their values based on whether they are default.

        Args:
            parameters_dict: The dictionary of parameters to process.
            defaulted_keys: A set of parameter keys that are considered to have default values.

        Returns:
            A dictionary of processed parameters with color coding and suffixes applied.
        """
        processed_params = {}
        for key, value in parameters_dict.items():
            processed_params[key] = self._format_parameter_with_color_and_suffix(
                value, key, defaulted_keys
            )
        return processed_params

    @abstractmethod
    def to_dict(self):
        pass


class PicselliaTrainingContext(PicselliaContext):
    def __init__(
        self,
        hyperparameters_cls: Type[HyperParameters],
        augmentation_parameters_cls: Type[AugmentationParameters],
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        organization_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "organization_name": self.organization_name,
                "experiment_id": self.experiment_id,
            },
            "hyperparameters": self._process_parameters(
                parameters_dict=self.hyperparameters.to_dict(),
                defaulted_keys=self.hyperparameters.defaulted_keys,
            ),
            "augmentation_parameters": self._process_parameters(
                parameters_dict=self.augmentation_parameters.to_dict(),
                defaulted_keys=self.augmentation_parameters.defaulted_keys,
            ),
        }

    def _initialize_experiment(self) -> Experiment:
        """Fetches the experiment from Picsellia using the experiment ID.

        The experiment, in a Picsellia training context,
        is the entity that contains all the information needed to train a model.

        Returns:
            The experiment fetched from Picsellia.
        """
        return self.client.get_experiment_by_id(self.experiment_id)


class PicselliaProcessingContext(PicselliaContext):
    def __init__(
        self,
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        organization_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
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
        self._model_version_id: Optional[str] = None
        self._input_dataset_version_id: Optional[str] = None
        self._output_dataset_version_id: Optional[str] = None

        self._initialize_context()

    @property
    def input_dataset_version_id(self) -> str:
        if not self._input_dataset_version_id:
            raise ValueError(
                "There's not input dataset version ID available. Please ensure the job is correctly configured."
            )
        return self._input_dataset_version_id

    @property
    def model_version_id(self) -> str:
        if not self._model_version_id:
            raise ValueError(
                "There's not model version ID available. Please ensure the job is correctly configured."
            )
        return self._model_version_id

    @property
    def output_dataset_version_id(self) -> str:
        if not self._output_dataset_version_id:
            raise ValueError(
                "There's not output dataset version ID available. Please ensure the job is correctly configured."
            )
        return self._output_dataset_version_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "job_id": self.job_id,
            },
            "model_version_id": self.model_version_id,
            "input_dataset_version_id": self.input_dataset_version_id,
            "output_dataset_version_id": self.output_dataset_version_id,
        }

    def _initialize_context(self) -> None:
        """Initializes the context by fetching the necessary information from the job."""
        context = self.job.sync()["dataset_version_processing_job"]

        self._model_version_id = context.get("model_version_id")
        self._input_dataset_version_id = context.get("input_dataset_version_id")
        self._output_dataset_version_id = context.get("output_dataset_version_id")

        if not self.model_version_id or not self.input_dataset_version_id:
            raise ValueError(
                "Failed to retrieve necessary context from the job. Please ensure the job is correctly configured."
            )

    def _initialize_job(self) -> picsellia.Job:
        """
        Fetches the job from Picsellia using the job ID.

        The Job, in a Picsellia processing context,
        is the entity that contains all the information needed to run a processing job.

        Returns:
            The job fetched from Picsellia.
        """
        return self.client.get_job_by_id(self.job_id)
