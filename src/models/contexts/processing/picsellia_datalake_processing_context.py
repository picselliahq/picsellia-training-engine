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
    """
    A context class designed for handling Picsellia datalake processing jobs.

    This class extends `PicselliaContext` to manage the specific setup and execution
    of a datalake processing job in Picsellia, including fetching job context,
    input/output datalakes, and model versions.

    Attributes:
        job_id (str): The job ID, either passed or fetched from environment variables.
        job (picsellia.Job): The Picsellia job object initialized from the job ID.
        job_type (str): The type of the job (e.g., pre-annotation).
        job_context (dict): The context of the job containing model version, datalakes, and other details.
        input_datalake (Datalake): The input datalake used in the job.
        output_datalake (Optional[Datalake]): The output datalake (optional).
        model_version (Optional[ModelVersion]): The model version associated with the job.
        data_ids (list[UUID]): List of data IDs fetched from the job payload.
        use_id (bool): A flag indicating whether to use data IDs.
        download_annotations (bool): A flag indicating whether to download annotations.
        processing_parameters (TParameters): The parameters used for the processing job.
    """

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
        """
        Initializes the PicselliaDatalakeProcessingContext with parameters to run a processing job.

        Args:
            processing_parameters_cls (Type[TParameters]): The class used to define the processing parameters.
            api_token (Optional[str], optional): The API token to authenticate with Picsellia. Defaults to None.
            host (Optional[str], optional): The host URL for the Picsellia platform. Defaults to None.
            organization_id (Optional[str], optional): The organization ID within Picsellia. Defaults to None.
            job_id (Optional[str], optional): The ID of the job to be processed. Defaults to None.
            use_id (Optional[bool], optional): Whether to use data IDs in the processing job. Defaults to True.
            download_annotations (Optional[bool], optional): Whether to download annotations for the datalake. Defaults to True.

        Raises:
            ValueError: If the job ID is not provided or found in the environment variables.
        """
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
        """
        Retrieves the model version ID if available, and ensures it is required for certain job types.

        Returns:
            Union[str, None]: The model version ID or None if not applicable.

        Raises:
            ValueError: If the model version ID is missing when it is required for the job type.
        """
        if (
            not self._model_version_id
            and self.job_type == ProcessingType.PRE_ANNOTATION
        ):
            raise ValueError(
                "Model version ID not found. Please ensure the job is correctly configured."
            )

        return self._model_version_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the current processing context to a dictionary format for easier serialization.

        Returns:
            Dict[str, Any]: A dictionary representation of the context, including job parameters,
            datalake IDs, and processing parameters.
        """
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
        """
        Initializes the job context by synchronizing the job data from Picsellia.

        Returns:
            Dict[str, Any]: The job context containing information such as input datalake, output datalake,
            model version ID, and other processing parameters.
        """
        job_context = self.job.sync()["datalake_processing_job"]

        return job_context

    def _initialize_job(self) -> picsellia.Job:
        """
        Initializes and retrieves the job from Picsellia using the job ID.

        Returns:
            picsellia.Job: The initialized job object fetched from Picsellia.
        """
        return self.client.get_job_by_id(self.job_id)

    def get_datalake(self, datalake_id: str) -> Datalake:
        """
        Fetches a datalake from Picsellia using the datalake ID.

        Args:
            datalake_id (str): The ID of the datalake to fetch.

        Returns:
            Datalake: The datalake object fetched from Picsellia.
        """
        return self.client.get_datalake(id=datalake_id)

    def get_model_version(self) -> ModelVersion:
        """
        Fetches the model version from Picsellia using the model version ID.

        Returns:
            ModelVersion: The model version object fetched from Picsellia.
        """
        return self.client.get_model_version_by_id(self.model_version_id)

    def get_data_ids(self) -> list[UUID]:
        """
        Retrieves the list of data IDs from the job's payload presigned URL.

        Returns:
            list[UUID]: A list of UUIDs representing the data IDs.

        Raises:
            ValueError: If the payload presigned URL is missing from the job context.
        """
        if self._payload_presigned_url:
            payload = requests.get(self._payload_presigned_url).json()
            data_ids = payload["data_ids"]
            uuid_data_ids = [UUID(data_id) for data_id in data_ids]
            return uuid_data_ids
        else:
            raise ValueError(
                "Payload presigned URL not found. Please ensure the job is correctly configured."
            )
