import os
from typing import Optional

from picsellia import DatasetVersion

from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.processing.processing_dataset_collection import (
    ProcessingDatasetCollection,
)


class ProcessingDatasetCollectionExtractor:
    def __init__(
        self,
        input_dataset_version: DatasetVersion,
        output_dataset_version: DatasetVersion,
        job_id: Optional[str] = None,
        use_id: Optional[bool] = True,
        download_annotations: Optional[bool] = True,
    ):
        self.input_dataset_version = input_dataset_version
        self.output_dataset_version = output_dataset_version
        if not job_id:
            self.destination_path = os.path.join(os.getcwd(), "current_job")
        else:
            self.destination_path = os.path.join(os.getcwd(), str(job_id))
        self.use_id = use_id
        self.download_annotations = download_annotations

    def get_dataset_collection(self) -> ProcessingDatasetCollection:
        # get input and output dataset versions
        input_dataset_context = DatasetContext(
            dataset_name="input",
            dataset_version=self.input_dataset_version,
            destination_path=self.destination_path,
            multi_asset=None,
            labelmap=None,
            use_id=self.use_id,
            download_annotations=self.download_annotations,
        )
        output_dataset_context = DatasetContext(
            dataset_name="output",
            dataset_version=self.output_dataset_version,
            destination_path=self.destination_path,
            multi_asset=None,
            labelmap=None,
            use_id=self.use_id,
            download_annotations=self.download_annotations,
        )
        return ProcessingDatasetCollection(
            input_dataset_context=input_dataset_context,
            output_dataset_context=output_dataset_context,
        )
