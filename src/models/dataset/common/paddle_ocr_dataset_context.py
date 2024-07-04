import os
from typing import Optional, Dict

from picsellia import DatasetVersion, Label
from picsellia.sdk.asset import MultiAsset

from src.models.dataset.common.dataset_context import DatasetContext


class PaddleOCRDatasetContext(DatasetContext):
    def __init__(
        self,
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
        self.paddle_ocr_bbox_annotations_path = os.path.join(
            self.destination_path,
            self.dataset_name,
            "annotations",
            "bbox",
            "annotations.txt",
        )
        self.paddle_ocr_text_annotations_path = os.path.join(
            self.destination_path,
            self.dataset_name,
            "annotations",
            "text",
            "annotations.txt",
        )
        self.text_image_dir = os.path.join(
            os.path.dirname(self.image_dir), "paddleocr_images"
        )
        os.makedirs(
            os.path.dirname(self.paddle_ocr_bbox_annotations_path), exist_ok=True
        )
        os.makedirs(
            os.path.dirname(self.paddle_ocr_text_annotations_path), exist_ok=True
        )
        os.makedirs(self.text_image_dir, exist_ok=True)
