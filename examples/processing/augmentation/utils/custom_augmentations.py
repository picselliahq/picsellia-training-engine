from typing import List, Dict
from PIL import Image
import numpy as np
from albumentations import (
    Compose,
    RandomBrightnessContrast,
    HorizontalFlip,
    ShiftScaleRotate,
    Blur,
    BboxParams,
)


def get_augmentation_pipeline() -> Compose:
    """
    Define and return the augmentation pipeline.

    Returns:
        Compose: An Albumentations Compose object with defined augmentations.
    """
    return Compose(
        [
            RandomBrightnessContrast(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            Blur(blur_limit=3, p=0.3),
        ],
        bbox_params=BboxParams(format="coco", label_fields=["id"]),
    )


def calculate_area(bbox: List[float]) -> float:
    """
    Calculate the area of a bounding box.

    Args:
        bbox (List[float]): Bounding box in [x_min, y_min, width, height] format.

    Returns:
        float: Area of the bounding box.
    """
    _, _, width, height = bbox
    return width * height


def build_annotation(annotation: Dict, bbox: List[float], area: float) -> Dict:
    """
    Build a new COCO annotation with updated bounding box and area.

    Args:
        annotation (Dict): Original annotation.
        bbox (List[float]): Updated bounding box.
        area (float): Recalculated area of the bounding box.

    Returns:
        Dict: Updated COCO annotation.
    """
    return {
        "id": annotation["id"],
        "image_id": annotation["image_id"],
        "category_id": annotation["category_id"],
        "bbox": bbox,
        "iscrowd": annotation.get("iscrowd", 0),
        "segmentation": annotation.get("segmentation", []),
        "area": area,
        "score": annotation.get("score", 0.0),
    }


def apply_augmentations(img: Image.Image, annotations: List[Dict], parameters):
    """
    Generate multiple augmented versions of the input image, updating annotations accordingly.

    Args:
        img (PIL.Image.Image): Input image.
        annotations (List[Dict]): List of COCO annotations for the image.
        parameters: Additional parameters for the augmentation pipeline.

    Returns:
        Tuple[List[Image.Image], List[List[Dict]]]: List of augmented images and their corresponding annotations.
    """
    # Convert PIL Image to NumPy array for Albumentations
    img_array = np.array(img)

    # Map annotations by ID for easy retrieval
    annotations_by_id = {ann["id"]: ann for ann in annotations}

    # Fetch the augmentation pipeline
    augmentation_pipeline = get_augmentation_pipeline()

    # Get the number of augmentations from parameters, defaulting to 3
    num_augmentations = parameters.num_augmentations

    # Generate multiple augmented images and annotations
    augmented_images = []
    augmented_annotations = []

    for _ in range(num_augmentations):  # Generate 3 augmentations
        augmented = augmentation_pipeline(
            image=img_array,
            bboxes=[ann["bbox"] for ann in annotations],
            id=[ann["id"] for ann in annotations],
        )

        # Convert augmented image back to PIL format
        augmented_img = Image.fromarray(augmented["image"])

        # Rebuild annotations with updated bounding boxes
        updated_annotations = [
            build_annotation(
                annotations_by_id[obj_id],
                bbox,
                calculate_area(bbox),
            )
            for bbox, obj_id in zip(augmented["bboxes"], augmented["id"])
        ]

        # Append results
        augmented_images.append(augmented_img)
        augmented_annotations.append(updated_annotations)

    return augmented_images, augmented_annotations
