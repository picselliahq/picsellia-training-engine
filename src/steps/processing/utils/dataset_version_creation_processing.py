from typing import List, Optional

from picsellia import Client, DatasetVersion, Data


class DatasetVersionCreationProcessing:
    def __init__(self, client: Client, output_dataset_version: DatasetVersion):
        self.client = client
        self.output_dataset_version = output_dataset_version

    def _upload_assets(
        self, images_to_upload: List[str], images_tags: Optional[List[str]] = None
    ) -> Data:
        datalake = self.client.get_datalake(name="default")
        data = datalake.upload_data(filepaths=images_to_upload, tags=images_tags)
        return data

    def _add_images_to_dataset_version(
        self, images_to_upload: List[str], images_tags: Optional[List[str]] = None
    ) -> None:
        data = self._upload_assets(
            images_to_upload=images_to_upload, images_tags=images_tags
        )
        adding_job = self.output_dataset_version.add_data(data=data)
        adding_job.wait_for_done()
