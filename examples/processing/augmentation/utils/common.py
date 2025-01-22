import importlib
import os
from copy import deepcopy
from glob import glob
from typing import Dict, List, Optional, Any

from PIL import Image

from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.coco_dataset_context import CocoDatasetContext


def get_image_id_by_filename(coco_data: Dict, filename: str) -> int:
    """
    Retrieve the image ID for a given filename.

    Args:
        coco_data (Dict): COCO dataset structure containing images.
        filename (str): Filename of the image.

    Returns:
        int: ID of the image.
    """
    for image in coco_data["images"]:
        if image["file_name"] == filename:
            return image["id"]
    raise ValueError(f"Image with filename '{filename}' not found.")


def get_annotations_for_image(coco_data: Dict, image_id: int) -> List[Dict]:
    """
    Retrieve all annotations associated with a given image ID.

    Args:
        coco_data (Dict): COCO dataset structure containing annotations.
        image_id (int): ID of the image.

    Returns:
        List[Dict]: List of annotations associated with the given image ID.
    """
    return [
        annotation
        for annotation in coco_data["annotations"]
        if annotation["image_id"] == image_id
    ]


def save_augmented_image(
    img: Image.Image, output_dir: str, base_filename: str, suffix: Optional[int] = None
) -> str:
    """
    Save an augmented image to the output directory.

    Args:
        img (Image.Image): Augmented image.
        output_dir (str): Directory to save the image.
        base_filename (str): Base filename for the image.
        suffix (int, optional): Suffix to append to the filename.

    Returns:
        str: Path to the saved image.
    """
    splited_filename = base_filename.split(".")
    extension = splited_filename[-1]
    if img.mode == "RGBA" and extension.lower() not in ["png", "tiff", "webp", "heic"]:
        extension = "png"
    base_filename_no_extension = ".".join(splited_filename[:-1])
    filename = (
        f"{base_filename_no_extension}_{suffix}.{extension}"
        if suffix
        else f"{base_filename_no_extension}.{extension}"
    )
    output_path = os.path.join(output_dir, filename)
    img.save(output_path)
    return filename


def update_coco_annotations(
    output_coco_data: Dict, annotations: List[Dict], new_image_id: int
) -> None:
    """
    Update COCO annotations with new image IDs and ensure unique annotation IDs.

    Args:
        output_coco_data (Dict): The COCO data structure to update.
        annotations (List[Dict]): List of annotations to update.
        new_image_id (int): The new image ID.

    Returns:
        None
    """
    for annotation in annotations:
        new_annotation = deepcopy(annotation)
        new_annotation["image_id"] = new_image_id
        new_annotation["id"] = len(output_coco_data["annotations"])  # Unique ID
        output_coco_data["annotations"].append(new_annotation)


def save_augmented_images_and_update_coco(
    augmented_images: List[Image.Image],
    augmented_annotations: List[List[Dict]],
    output_coco_data: Dict,
    output_dir: str,
    base_filename: str,
) -> None:
    """
    Save augmented images and update the COCO dataset metadata.

    Args:
        augmented_images (List[Image.Image]): List of augmented images.
        augmented_annotations (List[List[Dict]]): List of annotations for each augmented image.
        output_coco_data (Dict): The COCO dataset metadata to update.
        output_dir (str): Directory where augmented images are saved.
        base_filename (str): Base filename for the images.

    Returns:
        None
    """
    for i, (augmented_img, annotations) in enumerate(
        zip(augmented_images, augmented_annotations)
    ):
        # Add suffix to filename if there are multiple augmentations
        suffix = i if len(augmented_images) > 1 else None
        filename = save_augmented_image(
            augmented_img, output_dir, base_filename, suffix
        )

        # Create new image metadata
        new_image_id = len(output_coco_data["images"])
        output_coco_data["images"].append(
            {
                "id": new_image_id,
                "file_name": filename,
                "width": augmented_img.width,
                "height": augmented_img.height,
            }
        )

        # Update annotations for the new image
        update_coco_annotations(output_coco_data, annotations, new_image_id)


# Dynamically load `apply_augmentations` from the custom augmentations module
def load_apply_augmentations():
    """
    Dynamically loads the apply_augmentations function from the custom augmentations module.
    """
    module = importlib.import_module(
        "examples.processing.augmentation.utils.custom_augmentations"
    )
    return getattr(module, "apply_augmentations")


# Load the user-defined apply_augmentations function
apply_augmentations = load_apply_augmentations()


@step
def process_dataset(
    input_dataset: CocoDatasetContext, output_dataset: CocoDatasetContext
):
    """
    Apply augmentations to the input dataset images and save the augmented images in the output dataset.

    Args:
        input_dataset (object): Input dataset object with attributes like `images_dir` and `coco_data`.
        output_dataset (object): Output dataset object where augmented images and annotations are stored.
    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    if not input_dataset.images_dir:
        raise ValueError("Input dataset does not have images downloaded.")
    if not input_dataset.coco_data:
        raise ValueError("Input dataset does not have COCO data.")
    if not output_dataset.images_dir:
        raise ValueError("Output dataset does not have images directory.")

    # Initialize COCO structure for the output dataset
    output_coco_data: Dict[str, Any] = deepcopy(input_dataset.coco_data)
    output_coco_data["images"] = []
    output_coco_data["annotations"] = []

    # List all images in the input dataset
    image_paths = glob(os.path.join(input_dataset.images_dir, "*"))

    for image_path in image_paths:
        # Open the image
        img = Image.open(image_path).convert("RGB")

        # Get associated annotations
        image_filename = os.path.basename(image_path)
        img_id = get_image_id_by_filename(
            coco_data=input_dataset.coco_data, filename=image_filename
        )
        annotations = get_annotations_for_image(
            coco_data=input_dataset.coco_data, image_id=img_id
        )

        # Apply augmentations
        augmented_images, augmented_annotations = apply_augmentations(
            img=img, annotations=annotations, parameters=context.processing_parameters
        )

        # Save augmented images and update COCO metadata
        save_augmented_images_and_update_coco(
            augmented_images=augmented_images,
            augmented_annotations=augmented_annotations,
            output_coco_data=output_coco_data,
            output_dir=output_dataset.images_dir,
            base_filename=os.path.basename(image_path),
        )

    # Save updated COCO data
    output_dataset.coco_data = output_coco_data

    print(
        f"Processed {len(image_paths)} images and saved augmented images to {output_dataset.images_dir}."
    )
    return output_dataset
