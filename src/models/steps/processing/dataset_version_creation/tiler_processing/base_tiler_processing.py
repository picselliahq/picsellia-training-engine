import json
import logging
import math
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from picsellia import DatasetVersion
from picsellia.types.enums import InferenceType
from PIL import Image

from src.models.dataset.processing.processing_dataset_collection import (
    ProcessingDatasetCollection,
)

logger = logging.getLogger("picsellia-engine")


class TileMode(Enum):
    CONSTANT = "constant"
    DROP = "drop"
    REFLECT = "reflect"
    EDGE = "edge"
    WRAP = "wrap"


class BaseTilerProcessing(ABC):
    """Base class for tiler processing."""

    def __init__(
        self,
        tile_height: int,
        tile_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        min_annotation_area_ratio: float,
        min_annotation_width: Optional[float] = None,
        min_annotation_height: Optional[float] = None,
        tilling_mode: TileMode = TileMode.CONSTANT,
        constant_value: int = 114,
    ):
        self.tile_width = tile_width
        self.tile_height = tile_height

        self.overlap_width_ratio = overlap_width_ratio
        self.overlap_height_ratio = overlap_height_ratio

        self.min_annotation_area_ratio = 0
        self.min_annotation_width = 50
        self.min_annotation_height = 50

        self.tilling_mode = tilling_mode
        self.constant_value = constant_value

    @property
    def stride_x(self) -> int:
        return (
            self.tile_width
            if self.overlap_width_ratio == 0
            else int(self.tile_width * (1 - self.overlap_width_ratio))
        )

    @property
    def stride_y(self) -> int:
        return (
            self.tile_height
            if self.overlap_height_ratio == 0
            else int(self.tile_height * (1 - self.overlap_height_ratio))
        )

    def get_tile_coordinates_from_filename(self, tile_filename: str) -> Tuple[int, int]:
        """
        Extract tile coordinates from the filename.

        Args:
            tile_filename: Name of the tile file.

        Returns:
            Tuple containing the x and y coordinates of the tile.
        """
        parts = tile_filename.split(".")[0].split("_")
        return int(parts[-2]), int(parts[-1])

    def process_dataset_collection(
        self, dataset_collection: ProcessingDatasetCollection
    ) -> ProcessingDatasetCollection:
        """
        Process each dataset context of the dataset collection by tiling the images and annotations.

        Args:
            dataset_collection: The dataset collection to process.

        Returns:
            The updated dataset collection.
        """
        self._update_output_dataset_version_description_and_type(
            input_dataset_version=dataset_collection.input.dataset_version,
            output_dataset_version=dataset_collection.output.dataset_version,
        )

        self._process_dataset_collection(dataset_collection=dataset_collection)

        return dataset_collection

    def tile_image(self, image: Image.Image) -> np.ndarray:
        """
        Tile an input image into smaller overlapping tiles with various padding modes.

        Args:
            image (PIL.Image): The image to tile.

        Returns:
            np.ndarray: A batch of image tiles as a 4D numpy array (N, H, W, C).
        """

        image_array = np.array(image)

        n_tiles_x = math.ceil(image.width / self.stride_x)
        n_tiles_y = math.ceil(image.height / self.stride_y)

        tiles = []

        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                x1 = j * self.stride_x
                y1 = i * self.stride_y
                x2 = min(x1 + self.tile_width, image.width)
                y2 = min(y1 + self.tile_height, image.height)

                tile = image_array[y1:y2, x1:x2]

                if tile.shape[:2] != (self.tile_width, self.tile_height):
                    if self.tilling_mode == TileMode.DROP:
                        continue
                    else:
                        pad_width = [
                            (0, max(0, self.tile_height - tile.shape[0])),
                            (0, max(0, self.tile_width - tile.shape[1])),
                            (0, 0),
                        ]
                        pad_width = [pw for pw in pad_width if pw is not None]

                        if self.tilling_mode == TileMode.CONSTANT:
                            tile = np.pad(  # noqa
                                tile,
                                pad_width,
                                mode="constant",
                                constant_values=self.constant_value,
                            )
                        elif self.tilling_mode == TileMode.REFLECT:
                            tile = np.pad(tile, pad_width, mode="reflect")  # noqa
                        elif self.tilling_mode == TileMode.EDGE:
                            tile = np.pad(tile, pad_width, mode="edge")  # noqa
                        elif self.tilling_mode == TileMode.WRAP:
                            tile = np.pad(tile, pad_width, mode="wrap")  # noqa

                tiles.append(tile)

        tiles_data = np.array(tiles)
        return tiles_data

    def _load_coco_data(self, coco_file_path: str) -> Dict[str, Any]:
        """
        Load COCO data from a JSON file.

        Args:
            coco_file_path: Path to the COCO file.

        Returns:
            Dict[str, Any]: The COCO data extracted from the file.
        """
        with open(coco_file_path, "r") as f:
            return json.load(f)

    def _process_dataset_collection(
        self, dataset_collection: ProcessingDatasetCollection
    ) -> None:
        """
        Process all images and annotations from the dataset collection.

        This method handles different dataset types (classification, object detection, segmentation)
        and applies appropriate tiling strategies for each.

        Args:
            dataset_collection: A ProcessingDatasetCollection object containing input and output dataset information.

        Raises:
            ValueError: If the dataset type is not supported or configured.
        """
        dataset_type = dataset_collection.input.dataset_version.type

        if dataset_type == InferenceType.NOT_CONFIGURED:
            raise ValueError("Dataset type is not configured.")

        coco_data = self._load_coco_data(dataset_collection.input.coco_file_path)
        number_of_images = len(coco_data["images"])

        output_coco_data = {
            "images": [],
            "annotations": [],
            "categories": coco_data.get("categories", []),
        }

        logger.info(
            f"🔎 Starting the tiling processing for {number_of_images} images, "
            f"this may take a while depending on the number the size of the images."
        )

        current_tile_id = 0

        for idx, image_info in enumerate(coco_data["images"]):
            image_filename = image_info["file_name"]
            image_path = os.path.join(
                dataset_collection.input.image_dir, image_filename
            )
            image = Image.open(image_path)

            tiles_batch = self.tile_image(image=image)

            tiles_batch_info, current_tile_id, ignored_tiles_count = self._save_tiles(
                tiles_batch=tiles_batch,
                original_image_size=(image.width, image.height),
                original_filename=image_info["file_name"],
                current_tile_id=current_tile_id,
                output_dir=dataset_collection.output.image_dir,
                stride_x=self.stride_x,
                stride_y=self.stride_y,
                constant_value=self.constant_value,
            )

            self._tile_annotation(
                coco_data=coco_data,
                coco_image_info=image_info,
                output_coco_data=output_coco_data,
                tiles_batch_info=tiles_batch_info,
            )

            skipped_tiles_message = (
                f" ({ignored_tiles_count} tiles were skipped)"
                if ignored_tiles_count > 0
                else ""
            )

            logger.info(
                f"🖼️ Successfully tiled image {image_filename} ({idx + 1}/{number_of_images}) "
                f"in {tiles_batch.shape[0]} tiles"
                f"{skipped_tiles_message}"
            )

        self._save_coco_data(dataset_collection.output.coco_file_path, output_coco_data)
        dataset_collection.output.build_coco_file(
            coco_file_path=dataset_collection.output.coco_file_path
        )

    def _tile_annotation(
        self,
        coco_data: Dict[str, Any],
        coco_image_info: Dict[str, Any],
        output_coco_data: Dict[str, Any],
        tiles_batch_info: List[Dict[str, Any]],
    ) -> None:
        annotations = [
            annotation
            for annotation in coco_data["annotations"]
            if annotation["image_id"] == coco_image_info["id"]
        ]

        for tile_info in tiles_batch_info:
            output_coco_data["images"].append(tile_info)

            tile_x, tile_y = self.get_tile_coordinates_from_filename(
                tile_info["file_name"]
            )

            for annotation in annotations:
                new_annotation = self._adjust_coco_annotation(
                    annotation,
                    tile_x,
                    tile_y,
                    tile_info,
                    output_coco_data,
                )
                if new_annotation is not None:
                    output_coco_data["annotations"].append(new_annotation)

    def _save_coco_data(
        self, output_coco_path: str, output_coco_data: Dict[str, Any]
    ) -> None:
        """Save COCO data to a JSON file."""
        with open(output_coco_path, "w") as f:
            json.dump(output_coco_data, f)

    def _save_tiles(
        self,
        tiles_batch: np.ndarray,
        original_image_size: Tuple[int, int],
        original_filename: str,
        current_tile_id: int,
        output_dir: str,
        stride_x: int,
        stride_y: int,
        constant_value: int,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Save tiled images to the specified output directory using concurrent processing.

        Args:
            tiles_batch (np.ndarray): Numpy array of tiled images.
            original_image_size (Tuple[int, int]): Size of the original image (width, height).
            original_filename (str): Original filename of the image.
            current_tile_id (int): Current tile id to use for the tiles when writing the COCO data.
            output_dir (str): Directory to save the tiled images.
            stride_x (int): Horizontal stride used for tiling.
            stride_y (int): Vertical stride used for tiling.
            constant_value (int): Constant value used for padding.

        Returns:
            Tuple[List[Dict[str, Any]], int, int]: List of tile information, current tile id, and number of ignored tiles.
        """
        tile_infos: List[Dict[str, Any]] = []
        tiles_per_row = math.ceil(original_image_size[0] / stride_x)
        ignored_tiles_count = 0

        os.makedirs(output_dir, exist_ok=True)

        with ThreadPoolExecutor() as executor:
            future_to_tile = {}
            for idx, tile in enumerate(tiles_batch):
                if np.all(tile == constant_value):
                    ignored_tiles_count += 1
                    continue

                tile_y = (idx // tiles_per_row) * stride_y
                tile_x = (idx % tiles_per_row) * stride_x

                tile_filename = f"{os.path.splitext(original_filename)[0]}_tile_{tile_x}_{tile_y}.png"
                tile_path = os.path.join(output_dir, tile_filename)

                future = executor.submit(BaseTilerProcessing.save_tile, tile, tile_path)
                future_to_tile[future] = {
                    "file_name": tile_filename,
                    "width": tile.shape[1],
                    "height": tile.shape[0],
                    "id": current_tile_id,
                }

                current_tile_id += 1

            for future in as_completed(future_to_tile):
                tile_info = future_to_tile[future]
                try:
                    future.result()  # This will raise an exception if the tile saving failed
                    tile_infos.append(tile_info)
                except Exception as e:
                    print(f"Error saving tile {tile_info['file_name']}: {e}")

        return tile_infos, current_tile_id, ignored_tiles_count

    def _update_output_dataset_version_description_and_type(
        self,
        input_dataset_version: DatasetVersion,
        output_dataset_version: DatasetVersion,
    ) -> None:
        """
        Updates the output dataset version by adding a description and setting its type. The type of the output dataset
        version is the same as the type of the input dataset version.

        Args:
            input_dataset_version: The input dataset version.
            output_dataset_version: The output dataset version to update.
        """

        output_dataset_description = (
            f"Dataset sliced from dataset version "
            f"'{input_dataset_version.version}' "
            f"(id: {input_dataset_version.id}) "
            f"in dataset '{input_dataset_version.name}' "
            f"with slice size {self.tile_height}x{self.tile_width}."
        )

        output_dataset_version.update(
            description=output_dataset_description, type=input_dataset_version.type
        )

    @staticmethod
    def save_tile(tile: np.ndarray, tile_path: str) -> None:
        """
        Save a single tile as an image.

        Args:
            tile (np.ndarray): The tile image data.
            tile_path (str): The path where the tile should be saved.
        """
        tile_image = Image.fromarray(tile)
        tile_image.save(tile_path)

    @abstractmethod
    def _adjust_coco_annotation(
        self,
        annotation: Dict[str, Any],
        tile_x: int,
        tile_y: int,
        tile_info: Dict[str, Any],
        output_coco_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Adjust annotation for a specific tile.

        Args:
            annotation: Original annotation.
            tile_x: X-coordinate of the tile.
            tile_y: Y-coordinate of the tile.
            tile_info: Information about the tile in the output COCO data, usually containing the tile's id, width and height.
            output_coco_data: Output COCO data to which the adjusted annotation will be added.

        Returns:
            Optional[Dict[str, Any]]: Adjusted annotation for the tile.
        """
        pass
