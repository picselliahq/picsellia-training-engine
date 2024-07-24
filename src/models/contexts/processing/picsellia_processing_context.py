import os
from typing import Type, Optional, Any, Dict, Union, Generic

import picsellia  # type: ignore
from picsellia import DatasetVersion, ModelVersion
from picsellia.types.enums import ProcessingType
from src.models.contexts.common.picsellia_context import PicselliaContext, TParameters


class PicselliaProcessingContext(PicselliaContext, Generic[TParameters]):
    def __init__(
        self,
        processing_parameters_cls: Type[TParameters],
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        organization_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ):
        super().__init__(api_token, host, organization_id)

        self.job_id = job_id or os.environ.get("job_id")
        if not self.job_id:
            raise ValueError(
                "Job ID not provided. Please provide it as an argument or set the 'job_id' environment variable."
            )

        self.job = self._initialize_job()
        self.job_type = self.job.sync()["type"]

        self.job_context = self._initialize_job_context()

        self._model_version_id = self.job_context.get("model_version_id")
        self._input_dataset_version_id = self.job_context.get(
            "input_dataset_version_id"
        )
        self._output_dataset_version_id = self.job_context.get(
            "output_dataset_version_id"
        )

        self.input_dataset_version = self.get_dataset_version(
            self.input_dataset_version_id
        )
        if self._output_dataset_version_id:
            self.output_dataset_version = self.get_dataset_version(
                self._output_dataset_version_id
            )
        if self._model_version_id:
            self.model_version = self.get_model_version()

        self.processing_parameters = processing_parameters_cls(
            log_data=self.job_context["parameters"]
        )

    @property
    def input_dataset_version_id(self) -> str:
        if not self._input_dataset_version_id:
            raise ValueError(
                "There's not input dataset version ID available. Please ensure the job is correctly configured."
            )
        return self._input_dataset_version_id

    @property
    def model_version_id(self) -> Union[str, None]:
        if (
            not self._model_version_id
            and self.job_type == ProcessingType.PRE_ANNOTATION
        ):
            raise ValueError(
                "Model version ID not found. Please ensure the job is correctly configured."
            )

        return self._model_version_id

    @property
    def output_dataset_version_id(self) -> Union[str, None]:
        if not self._output_dataset_version_id:
            if self.job_type == ProcessingType.DATASET_VERSION_CREATION:
                raise ValueError(
                    "Output dataset version ID not found. Please ensure the job is correctly configured."
                )
            else:
                self._output_dataset_version_id = self._input_dataset_version_id
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
            "processing_parameters": self._process_parameters(
                parameters_dict=self.processing_parameters.to_dict(),
                defaulted_keys=self.processing_parameters.defaulted_keys,
            ),
        }

    def _initialize_job_context(self) -> Dict[str, Any]:
        """Initializes the context by fetching the necessary information from the job."""
        job_context = self.job.sync()["dataset_version_processing_job"]

        return job_context

    def _initialize_job(self) -> picsellia.Job:
        """
        Fetches the job from Picsellia using the job ID.

        The Job, in a Picsellia processing context,
        is the entity that contains all the information needed to run a processing job.

        Returns:
            The job fetched from Picsellia.
        """
        return self.client.get_job_by_id(self.job_id)

    def get_dataset_version(self, dataset_version_id: str) -> DatasetVersion:
        """
        Fetches the dataset version from Picsellia using the input dataset version ID.

        The DatasetVersion, in a Picsellia processing context,
        is the entity that contains all the information needed to process a dataset.

        Returns:
            The dataset version fetched from Picsellia.
        """
        return self.client.get_dataset_version_by_id(dataset_version_id)

    def get_model_version(self) -> ModelVersion:
        """
        Fetches the model version from Picsellia using the model version ID.

        The ModelVersion, in a Picsellia processing context,
        is the entity that contains all the information needed to process a model.

        Returns:
            The model version fetched from Picsellia.
        """
        return self.client.get_model_version_by_id(self.model_version_id)
