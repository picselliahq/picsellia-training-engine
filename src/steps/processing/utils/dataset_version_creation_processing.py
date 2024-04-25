from typing import List, Optional, Tuple

from picsellia import Client, DatasetVersion, Data
from picsellia.services.error_manager import ErrorManager
from picsellia.types.enums import InferenceType


class DatasetVersionCreationProcessing:
    def __init__(
        self,
        client: Client,
        output_dataset_version: DatasetVersion,
        dataset_type: InferenceType,
        dataset_description: str,
        datalake_name: str = "default",
    ):
        self.client = client
        self.output_dataset_version = output_dataset_version
        self.output_dataset_version.update(
            description=dataset_description, type=dataset_type
        )
        self.datalake = self.client.get_datalake(name=datalake_name)

    def _upload_data_with_error_manager(
        self, images_to_upload: List[str], images_tags: Optional[List[str]] = None
    ) -> Tuple[Data, List[str]]:
        error_manager = ErrorManager()
        data = self.datalake.upload_data(
            filepaths=images_to_upload, tags=images_tags, error_manager=error_manager
        )
        error_paths = [error.path for error in error_manager.errors]
        return data, error_paths

    def _upload_images_to_datalake(
        self, images_to_upload: List[str], images_tags: Optional[List[str]] = None
    ) -> List[Data]:
        all_uploaded_data = []
        uploaded_data, error_paths = self._upload_data_with_error_manager(
            images_to_upload=images_to_upload, images_tags=images_tags
        )
        all_uploaded_data.append(uploaded_data)
        while error_paths:
            uploaded_data, error_paths = self._upload_data_with_error_manager(
                images_to_upload=error_paths, images_tags=images_tags
            )
            all_uploaded_data.append(uploaded_data)
        return all_uploaded_data

    def _add_images_to_dataset_version(
        self, images_to_upload: List[str], images_tags: Optional[List[str]] = None
    ) -> None:
        data = self._upload_images_to_datalake(
            images_to_upload=images_to_upload, images_tags=images_tags
        )
        adding_job = self.output_dataset_version.add_data(data=data)
        adding_job.wait_for_done()
