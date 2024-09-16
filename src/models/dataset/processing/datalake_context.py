import logging
import os
from typing import Optional, List
from uuid import UUID

from picsellia import Datalake
logger = logging.getLogger(__name__)


class DatalakeContext:

    def __init__(
        self,
        datalake_name: str,
        datalake: Datalake,
        destination_path: str,
        data_ids: Optional[List[UUID]] = None,
        use_id: Optional[bool] = True,
    ):
        self.datalake_name = datalake_name
        self.datalake = datalake
        self.data_ids = data_ids
        self.destination_path = destination_path
        self.use_id = use_id

        self._initialize_paths()

    def _initialize_paths(self):
        self.dataset_path = os.path.join(self.destination_path, self.datalake_name)
        self.image_dir = os.path.join(self.dataset_path, "images")

    def download_data(self, image_dir: str) -> None:
        os.makedirs(image_dir, exist_ok=True)
        if self.data_ids:
            data = self.datalake.list_data(ids=self.data_ids)
            data.download(target_path=image_dir, use_id=self.use_id)
