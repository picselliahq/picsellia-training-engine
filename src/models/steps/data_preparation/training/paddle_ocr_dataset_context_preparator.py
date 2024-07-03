import json
import os
from typing import Union, Dict, List

from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.dataset.common.paddle_ocr_dataset_context import PaddleOCRDatasetContext


def write_class_file(coco_data: Dict, class_file_path: str):
    categories = [category["name"] for category in coco_data["categories"]]
    with open(class_file_path, "w", encoding="utf-8") as file:
        for category in sorted(categories):
            file.write(category.upper() + "\n")


def write_annotations_file(data, output_path):
    with open(output_path, "w") as file:
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
            print(f'Found image id {image["id"]} for image {image_filename}')
            return image["id"]
    return None


def get_points_from_bbox(bbox: List[int]) -> List[List[int]]:
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def load_coco_text(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        return json.load(file)


def process_annotations(coco: Dict, image_directory: str):
    processed_data: List[str] = []
    paddle_ocr_annotations: List[Dict] = []
    group_image_id = None
    for annotation in coco["annotations"]:
        print(f'Processing annotation {annotation["id"]}')
        current_image_id = annotation["image_id"]
        print(f"Current image id: {current_image_id}")
        if not group_image_id:
            group_image_id = current_image_id
        if group_image_id and current_image_id != group_image_id:
            image_path = os.path.join(
                image_directory, coco["images"][group_image_id]["file_name"]
            )
            processed_data.append(f"{image_path}\t{json.dumps(paddle_ocr_annotations)}")
            paddle_ocr_annotations = []
            group_image_id = current_image_id
        else:
            paddle_ocr_annotation = {
                "transcription": annotation["utf8_string"],
                "label": find_category_name(
                    coco["categories"], annotation["category_id"]
                ),
                "points": get_points_from_bbox(annotation["bbox"]),
                "id": annotation["id"],
                "linking": annotation.get("linking", []),
            }
            paddle_ocr_annotations.append(paddle_ocr_annotation)
    return processed_data


class PaddleOCRDatasetContextPreparator:
    def __init__(self, dataset_context: TDatasetContext):
        """
        Initializes the organizer with a given dataset context.

        Args:
            dataset_context (DatasetContext): The dataset context to organize.
        """
        self.dataset_context = dataset_context
        self.destination_path = dataset_context.destination_path
        self.paddle_ocr_dataset_context = PaddleOCRDatasetContext(
            paddle_ocr_annotations_path=os.path.join(
                self.destination_path, f"{self.dataset_context.dataset_name}.txt"
            ),
            paddle_ocr_class_path=os.path.join(
                self.destination_path,
                f"{self.dataset_context.dataset_name}_class_list.txt",
            ),
            dataset_name=self.dataset_context.dataset_name,
            dataset_version=self.dataset_context.dataset_version,
            destination_path=self.dataset_context.destination_path,
            multi_asset=self.dataset_context.multi_asset,
            labelmap=self.dataset_context.labelmap,
            use_id=self.dataset_context.use_id,
        )

    def organize(self) -> PaddleOCRDatasetContext:
        coco_data = load_coco_text(self.dataset_context.coco_file_path)
        paddleocr_annotations = process_annotations(
            coco_data, self.dataset_context.image_dir
        )
        write_annotations_file(
            paddleocr_annotations,
            self.paddle_ocr_dataset_context.paddle_ocr_annotations_path,
        )
        write_class_file(
            coco_data, self.paddle_ocr_dataset_context.paddle_ocr_class_path
        )
        return self.paddle_ocr_dataset_context
