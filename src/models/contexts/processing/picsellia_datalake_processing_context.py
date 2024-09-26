import os
from typing import Type, Optional, Any, Dict, Union, Generic
from uuid import UUID

import picsellia  # type: ignore
from picsellia import ModelVersion, Datalake
from picsellia.types.enums import ProcessingType

from src.models.contexts.common.picsellia_context import PicselliaContext
from src.models.parameters.common.parameters import TParameters

import requests


class PicselliaDatalakeProcessingContext(PicselliaContext, Generic[TParameters]):
    def __init__(
        self,
        processing_parameters_cls: Type[TParameters],
        api_token: Optional[str] = None,
        host: Optional[str] = None,
        organization_id: Optional[str] = None,
        job_id: Optional[str] = None,
        use_id: Optional[bool] = True,
        download_annotations: Optional[bool] = True,
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
        self._input_datalake_id = self.job_context.get("input_datalake_id")
        self._output_datalake_id = self.job_context.get("output_datalake_id")
        self._payload_presigned_url = self.job_context.get("payload_presigned_url")

        if self._input_datalake_id:
            self.input_datalake = self.get_datalake(self._input_datalake_id)
        else:
            raise ValueError(
                "Input datalake ID not found. Please ensure the job is correctly configured."
            )
        if self._output_datalake_id:
            self.output_datalake = self.get_datalake(self._output_datalake_id)
        else:
            self.output_datalake = None
        if self._model_version_id:
            self.model_version = self.get_model_version()
        else:
            self.model_version = None
        self.data_ids = self.get_data_ids()

        self.use_id = use_id
        self.download_annotations = download_annotations

        self.processing_parameters = processing_parameters_cls(
            log_data=self.job_context["parameters"]
        )

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "job_id": self.job_id,
            },
            "model_version_id": self.model_version_id,
            "input_datalake_id": self._input_datalake_id,
            "output_datalake_id": self._output_datalake_id,
            "processing_parameters": self._process_parameters(
                parameters_dict=self.processing_parameters.to_dict(),
                defaulted_keys=self.processing_parameters.defaulted_keys,
            ),
        }

    def _initialize_job_context(self) -> Dict[str, Any]:
        """Initializes the context by fetching the necessary information from the job."""
        job_context = self.job.sync()["datalake_processing_job"]

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

    def get_datalake(self, datalake_id: str) -> Datalake:
        """
        Fetches the datalake from Picsellia using the datalake ID.

        The Datalake, in a Picsellia processing context,
        is the entity that contains all the data needed to process a model.

        Returns:
            The datalake fetched from Picsellia.
        """
        return self.client.get_datalake(id=datalake_id)

    def get_model_version(self) -> ModelVersion:
        """
        Fetches the model version from Picsellia using the model version ID.

        The ModelVersion, in a Picsellia processing context,
        is the entity that contains all the information needed to process a model.

        Returns:
            The model version fetched from Picsellia.
        """
        return self.client.get_model_version_by_id(self.model_version_id)

    def get_data_ids(self) -> list[UUID]:
        if self._payload_presigned_url:
            payload = requests.get(self._payload_presigned_url).json()
            data_ids = payload["data_ids"]
            uuid_data_ids = [UUID(data_id) for data_id in data_ids]
            return uuid_data_ids
        else:
            raise ValueError(
                "Payload presigned URL not found. Please ensure the job is correctly configured."
            )
