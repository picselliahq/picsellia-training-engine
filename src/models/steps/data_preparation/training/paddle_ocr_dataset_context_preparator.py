import json
import os
from typing import Union, Dict, List

import numpy as np

from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext

from imutils import perspective
import cv2


def write_annotations_file(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        for line in data:
            file.write(line + "\n")


def find_category_id(categories: List[Dict], category_name: str) -> Union[str, None]:
    for category in categories:
        if category["name"] == category_name:
            return category["id"]
    return None


def find_category_name(categories: List[Dict], category_id: str) -> Union[str, None]:
    for category in categories:
        if category["id"] == category_id:
            return category["name"]
    return None


def find_image_id(images: List[Dict], image_filename: str) -> Union[str, None]:
    for image in images:
        if image["file_name"] == image_filename:
            return image["id"]
    return None


def get_points_from_bbox(bbox: List[int]) -> List[List[int]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def load_coco_text(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        return json.load(file)


def get_bbox_annotations(coco: Dict, image_directory: str):
    processed_data: List[str] = []
    paddle_ocr_annotations: List[Dict] = []
    group_image_id = None

    def append_annotations():
        if group_image_id is not None:
            image_path = os.path.join(
                image_directory, coco["images"][group_image_id]["file_name"]
            )
            processed_data.append(
                f"{image_path}\t{json.dumps(paddle_ocr_annotations, ensure_ascii=False)}"
            )

    for annotation in coco["annotations"]:
        current_image_id = annotation["image_id"]
        if group_image_id is None:
            group_image_id = current_image_id

        if current_image_id != group_image_id:
            append_annotations()
            paddle_ocr_annotations = []
            group_image_id = current_image_id

        paddle_ocr_annotation = {
            "transcription": find_category_name(
                coco["categories"], annotation["category_id"]
            ),
            "points": get_points_from_bbox(annotation["bbox"]),
        }
        paddle_ocr_annotations.append(paddle_ocr_annotation)

    append_annotations()

    return processed_data


def get_text_annotations(coco: Dict, image_directory: str, new_image_directory: str):
    processed_data: List[str] = []
    img_counter = 0

    for annotation in coco["annotations"]:
        current_image_id = annotation["image_id"]
        image_path = os.path.join(
            image_directory, coco["images"][current_image_id]["file_name"]
        )

        points = get_points_from_bbox(annotation["bbox"])

        formatted_points = np.asarray([(x, y) for x, y in points], dtype=np.float32)
        image = cv2.imread(image_path)
        warped = perspective.four_point_transform(image, formatted_points)
        new_image_filename = "img_" + str(img_counter) + ".png"
        new_image_path = os.path.join(new_image_directory, new_image_filename)
        cv2.imwrite(new_image_path, warped)

        processed_data.append(f"{new_image_path}\t{annotation['utf8_string']}")
        img_counter = img_counter + 1

    return processed_data


class PaddleOCRDatasetContextPreparator:
    def __init__(self, dataset_context: TDatasetContext):
        """
        Initializes the organizer with a given dataset context.

        Args:
            dataset_context (DatasetContext): The dataset context to organize.
        """
        self.dataset_context = dataset_context
        self.paddle_ocr_dataset_context = PaddleOCRDatasetContext(
            dataset_name=self.dataset_context.dataset_name,
            dataset_version=self.dataset_context.dataset_version,
            destination_path=self.dataset_context.destination_path,
            multi_asset=self.dataset_context.multi_asset,
            labelmap=self.dataset_context.labelmap,
            use_id=self.dataset_context.use_id,
        )

    def organize(self) -> PaddleOCRDatasetContext:
        coco_data = load_coco_text(self.dataset_context.coco_file_path)
        paddleocr_bbox_annotations = get_bbox_annotations(
            coco=coco_data, image_directory=self.dataset_context.image_dir
        )
        paddleocr_text_annotations = get_text_annotations(
            coco=coco_data,
            image_directory=self.dataset_context.image_dir,
            new_image_directory=self.paddle_ocr_dataset_context.text_image_dir,
        )

        write_annotations_file(
            paddleocr_bbox_annotations,
            self.paddle_ocr_dataset_context.paddle_ocr_bbox_annotations_path,
        )
        write_annotations_file(
            paddleocr_text_annotations,
            self.paddle_ocr_dataset_context.paddle_ocr_text_annotations_path,
        )
        return self.paddle_ocr_dataset_context
