import os
from typing import Optional, Dict

from picsellia import DatasetVersion, Label
from picsellia.sdk.asset import MultiAsset

from src.models.dataset.common.dataset_context import DatasetContext


class PaddleOCRDatasetContext(DatasetContext):
    def __init__(
        self,
        paddle_ocr_annotations_path: str,
        paddle_ocr_class_path: str,
        dataset_name: str,
        dataset_version: DatasetVersion,
        destination_path: str,
        multi_asset: Optional[MultiAsset] = None,
        labelmap: Optional[Dict[str, Label]] = None,
        use_id: Optional[bool] = True,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            destination_path=destination_path,
            multi_asset=multi_asset,
            labelmap=labelmap,
            use_id=use_id,
        )
        self.paddle_ocr_annotations_path = paddle_ocr_annotations_path
        self.paddle_ocr_class_path = paddle_ocr_class_path
        os.makedirs(os.path.dirname(self.paddle_ocr_annotations_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.paddle_ocr_class_path), exist_ok=True)
