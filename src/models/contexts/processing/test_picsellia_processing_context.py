from typing import Any, Dict, Optional
from src.models.contexts.common.picsellia_context import PicselliaContext
from picsellia.types.enums import ProcessingType
from picsellia import DatasetVersion, ModelVersion
from picsellia.types.enums import ProcessingType


class TestPicselliaProcessingContext(PicselliaContext):
    """
    This class is used to test a processing pipeline without a real job execution on Picsellia (without giving a real job ID).
    """

    def __init__(
            self,
            api_token: Optional[str] = None,
            host: Optional[str] = None,
            organization_id: Optional[str] = None,
            job_id: Optional[str] = None,
            job_type: Optional[ProcessingType] = None,
            input_dataset_version_id: Optional[str] = None,
            output_dataset_version_id: Optional[str] = None,
            model_version_id: Optional[str] = None,
            processing_parameters=None,
    ):
        # Initialize the Picsellia client from the base class
        super().__init__(api_token, host, organization_id)
        self.job_id = job_id
        self.job_type = job_type
        self.input_dataset_version_id = input_dataset_version_id
        self.output_dataset_version_id = output_dataset_version_id
        self.model_version_id = model_version_id
        print(self.model_version_id)
        if self.input_dataset_version_id:
            self.input_dataset_version = self.get_dataset_version(
                self.input_dataset_version_id
            )
        if self.output_dataset_version_id:
            self.output_dataset_version = self.get_dataset_version(
                self.output_dataset_version_id
            )
        if self.model_version_id:
            self.model_version = self.get_model_version()
        self.processing_parameters = processing_parameters

    def get_dataset_version(self, dataset_version_id) -> DatasetVersion:
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_parameters": {
                "host": self.host,
                "organization_id": self.organization_id,
                "job_type": self.job_type,
                "input_dataset_version_id": self.input_dataset_version_id,
                "output_dataset_version_id": self.output_dataset_version_id,
                "model_version_id": self.model_version_id,
            },
            "processing_parameters": self.processing_parameters,
        }
