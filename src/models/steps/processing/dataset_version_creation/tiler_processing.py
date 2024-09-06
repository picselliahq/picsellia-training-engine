import json
import logging
import math
import os
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


class TilerProcessing:
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.
    """

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

        self.min_annotation_area_ratio = min_annotation_area_ratio
        self.min_annotation_width = min_annotation_width
        self.min_annotation_height = min_annotation_height

        self.tilling_mode = tilling_mode
        self.constant_value = constant_value

        self.current_tile_id = 0

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

    def process_dataset_collection(
        self, dataset_collection: ProcessingDatasetCollection
    ) -> ProcessingDatasetCollection:
        """
        Processes a dataset collection to tile the images and annotations.

        Args:
            dataset_collection: The dataset collection to process.

        Returns:
            The processed dataset collection.
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
        logger.info(f"🖼️ Successfully tiled image with shape {tiles_data.shape}")

        return tiles_data

    def _adjust_coco_annotation(
        self,
        annotation: Dict[str, Any],
        tile_x: int,
        tile_y: int,
        tile_info: Dict[str, Any],
        output_coco_data: Dict[str, Any],
        dataset_type: InferenceType,
    ) -> Optional[Dict[str, Any]]:
        """
        Adjust annotation for a specific tile.

        Args:
            annotation: Original annotation.
            tile_x: X-coordinate of the tile.
            tile_y: Y-coordinate of the tile.
            tile_info: Information about the tile.
            output_coco_data: Dict to store the processed COCO data.
            dataset_type: Type of the dataset.

        Returns:
            Adjusted annotation or None if the annotation doesn't intersect with the tile.
        """
        bbox = annotation["bbox"]
        x, y, width, height = bbox

        # Calculate the intersection of the bounding box with the tile
        ix1 = max(x, tile_x)
        iy1 = max(y, tile_y)
        ix2 = min(x + width, tile_x + tile_info["width"])
        iy2 = min(y + height, tile_y + tile_info["height"])

        # Check if there's an intersection
        if ix2 <= ix1 or iy2 <= iy1:
            return None

        # Calculate new bounding box coordinates relative to the tile
        new_bbox_x = ix1 - tile_x
        new_bbox_y = iy1 - tile_y
        new_bbox_width = ix2 - ix1
        new_bbox_height = iy2 - iy1

        new_bbox = [new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height]
        new_bbox_area = new_bbox_width * new_bbox_height

        # Check if the intersection area is large enough
        if not self._should_annotation_be_kept(bbox, new_bbox):
            return None

        # Create new annotation
        new_annotation = annotation.copy()
        new_annotation["id"] = len(output_coco_data["annotations"])
        new_annotation["image_id"] = tile_info["id"]
        new_annotation["bbox"] = new_bbox
        new_annotation["area"] = new_bbox_area

        # Handle segmentation for segmentation datasets
        if dataset_type == InferenceType.SEGMENTATION and "segmentation" in annotation:
            new_segmentation = self._adjust_segmentation_annotation(
                polygons=annotation["segmentation"],
                tile_x=tile_x,
                tile_y=tile_y,
                tile_width=tile_info["width"],
                tile_height=tile_info["height"],
            )
            if not new_segmentation:
                return None
            new_annotation["segmentation"] = new_segmentation

        return new_annotation

    def _adjust_segmentation_annotation(
        self,
        polygons: List[List[float]],
        tile_x: int,
        tile_y: int,
        tile_width: int,
        tile_height: int,
    ) -> List[List[float]]:
        """
        Adjust segmentation coordinates for a tile and clip the polygon to the tile boundaries.

        Args:
            polygons (List[List[float]]): Original segmentation coordinates.
            tile_x (int): X-coordinate of the tile's top-left corner.
            tile_y (int): Y-coordinate of the tile's top-left corner.
            tile_width (int): Width of the tile.
            tile_height (int): Height of the tile.

        Returns:
            List[List[float]]: Adjusted and clipped segmentation coordinates.
        """

        def clip_polygon(polygon, clip_rect):
            """Clip a polygon to a rectangle using Sutherland-Hodgman algorithm."""

            def inside(p, cp1, cp2):
                return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (
                    p[0] - cp1[0]
                )

            def compute_intersection(cp1, cp2, s, e):
                dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
                dp = [s[0] - e[0], s[1] - e[1]]
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = s[0] * e[1] - s[1] * e[0]
                n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
                return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

            output_list = polygon
            cp1 = clip_rect[-1]

            for clip_vertex in clip_rect:
                cp2 = clip_vertex
                input_list = output_list
                output_list = []

                if not input_list:
                    break

                s = input_list[-1]

                for subject_vertex in input_list:
                    e = subject_vertex
                    if inside(e, cp1, cp2):
                        if not inside(s, cp1, cp2):
                            output_list.append(compute_intersection(cp1, cp2, s, e))
                        output_list.append(e)
                    elif inside(s, cp1, cp2):
                        output_list.append(compute_intersection(cp1, cp2, s, e))
                    s = e
                cp1 = cp2

            return output_list

        adjusted_segmentation = []
        clip_rect = [
            (0, 0),
            (tile_width, 0),
            (tile_width, tile_height),
            (0, tile_height),
        ]

        for polygon in polygons:
            # Convert flat list to list of points
            points = [
                (polygon[i] - tile_x, polygon[i + 1] - tile_y)
                for i in range(0, len(polygon), 2)
            ]

            # Clip the polygon
            clipped_poly = clip_polygon(points, clip_rect)

            # Convert back to flat list and add to adjusted segmentation
            if (
                len(clipped_poly) > 2
            ):  # Only add if the polygon is valid (at least 3 points)
                adjusted_poly = [coord for point in clipped_poly for coord in point]
                adjusted_segmentation.append(adjusted_poly)

        return adjusted_segmentation

    def _get_tile_coordinates_from_filename(
        self, tile_filename: str
    ) -> Tuple[int, int]:
        """
        Extract tile coordinates from the filename.

        Args:
            tile_filename: Name of the tile file.

        Returns:
            Tuple containing the x and y coordinates of the tile.
        """
        parts = tile_filename.split(".")[0].split("_")
        return int(parts[-2]), int(parts[-1])

    def _load_coco_data(self, coco_file_path: str) -> Dict[str, Any]:
        """
        Load COCO data from a JSON file.

        Args:
            coco_file_path: Path to the COCO JSON file.

        Returns:
            Dict containing the COCO data.
        """
        with open(coco_file_path, "r") as f:
            return json.load(f)

    def _process_classification_image(
        self,
        annotation: Dict[str, Any],
        tiles_batch_info: List[Dict[str, Any]],
        output_coco_data: Dict[str, Any],
    ) -> None:
        """
        Process classification images for tiling.

        Args:
            annotation: Original classification annotation.
            tiles_batch_info: List of dictionaries containing information about each tile.
            output_coco_data: Dict to store the processed COCO data.
        """
        for tile_info in tiles_batch_info:
            # Add image information
            image_info = {
                "id": tile_info["id"],
                "file_name": tile_info["file_name"],
                "width": tile_info["width"],
                "height": tile_info["height"],
            }
            output_coco_data["images"].append(image_info)

            annotation = {
                "id": len(output_coco_data["annotations"]),
                "image_id": tile_info["id"],
                "category_id": annotation["category_id"],
                "bbox": [],
                "segmentation": [],
                "area": 0,
                "iscrowd": 0,
            }
            output_coco_data["annotations"].append(annotation)

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

        output_coco_data = {
            "images": [],
            "annotations": [],
            "categories": coco_data.get("categories", []),
        }

        for image_info in coco_data["images"]:
            image_path = os.path.join(
                dataset_collection.input.image_dir, image_info["file_name"]
            )
            image = Image.open(image_path)

            tiles_batch = self.tile_image(image=image)

            tiles_batch_info = self._save_tiles(
                original_image_size=(image.width, image.height),
                tiles_batch=tiles_batch,
                original_filename=image_info["file_name"],
                output_dir=dataset_collection.output.image_dir,
            )

            self._tile_annotation(
                coco_data=coco_data,
                coco_image_info=image_info,
                output_coco_data=output_coco_data,
                tiles_batch_info=tiles_batch_info,
                dataset_type=dataset_type,
            )

        self._save_coco_data(dataset_collection.output.coco_file_path, output_coco_data)
        dataset_collection.output.build_coco_file(
            coco_file_path=dataset_collection.output.coco_file_path
        )

    def _process_object_detection_and_segmentation_annotations(
        self,
        annotations: List[Dict[str, Any]],
        tiles_batch_info: List[Dict[str, Any]],
        output_coco_data: Dict[str, Any],
        dataset_type: InferenceType,
    ) -> None:
        """
        Process object detection and segmentation annotations for tiled images.

        Args:
            annotations: List of annotations for the original image.
            tiles_batch_info: List of dictionaries containing information about each tile.
            output_coco_data: Dict to store the processed COCO data.
            dataset_type: Type of the dataset (OBJECT_DETECTION or SEGMENTATION).
        """
        for tile_info in tiles_batch_info:
            output_coco_data["images"].append(tile_info)

            tile_x, tile_y = self._get_tile_coordinates_from_filename(
                tile_info["file_name"]
            )

            for annotation in annotations:
                new_annotation = self._adjust_coco_annotation(
                    annotation,
                    tile_x,
                    tile_y,
                    tile_info,
                    output_coco_data,
                    dataset_type,
                )
                if new_annotation:
                    output_coco_data["annotations"].append(new_annotation)

    def _save_coco_data(
        self, output_coco_path: str, output_coco_data: Dict[str, Any]
    ) -> None:
        """
        Save COCO data to a JSON file.

        Args:
            output_coco_path: Path where the output COCO JSON file will be saved.
            output_coco_data: Dict containing the COCO data to be saved.
        """
        with open(output_coco_path, "w") as f:
            json.dump(output_coco_data, f)

    def _save_tiles(
        self,
        original_image_size: Tuple[int, int],
        tiles_batch: np.ndarray,
        original_filename: str,
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        """
        Save tiled images to the specified output directory.

        Args:
            original_image_size: Size of the original image.
            tiles_batch: Numpy array of tiled images.
            original_filename: Original filename of the image.
            output_dir: Directory to save the tiled images.

        Returns:
            List of dictionaries containing information about each saved tile.
        """
        tile_infos: List[Dict[str, Any]] = []

        tiles_per_row = math.ceil(original_image_size[0] / self.stride_x)

        for idx, tile in enumerate(tiles_batch):
            # Skip the tiles that are only made of the constant value
            if np.all(tile == self.constant_value):
                continue

            tile_y = (idx // tiles_per_row) * self.stride_y
            tile_x = (idx % tiles_per_row) * self.stride_x

            tile_filename = (
                f"{os.path.splitext(original_filename)[0]}_tile_{tile_x}_{tile_y}.png"
            )
            tile_path = os.path.join(output_dir, tile_filename)

            tile_image = Image.fromarray(tile)
            tile_image.save(tile_path)

            tile_infos.append(
                {
                    "file_name": tile_filename,
                    "width": tile.shape[1],
                    "height": tile.shape[0],
                    "id": self.current_tile_id,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                }
            )

            self.current_tile_id += 1

        return tile_infos

    def _should_annotation_be_kept(
        self, old_bounding_box: List[int], new_bounding_box: List[int]
    ) -> bool:
        """
        Check if the annotation should be kept based on the new bounding box.

        Args:
            old_bounding_box: Original bounding box.
            new_bounding_box: New bounding box.

        Returns:
            True if the annotation should be kept, False otherwise.
        """
        old_x, old_y, old_width, old_height = old_bounding_box
        new_x, new_y, new_width, new_height = new_bounding_box

        # Check if the new bounding box is large enough
        if new_width < old_width:
            if (
                self.min_annotation_width is not None
                and new_width < self.min_annotation_width
            ):
                return False

        # Check if the new bounding box is tall enough
        if new_height < old_height:
            if (
                self.min_annotation_height is not None
                and new_height < self.min_annotation_height
            ):
                return False

        # Check the area as well
        old_area = old_width * old_height
        new_area = new_width * new_height

        if new_area / old_area < self.min_annotation_area_ratio:
            return False

        return True

    def _tile_annotation(
        self,
        coco_data: Dict[str, Any],
        coco_image_info: Dict[str, Any],
        output_coco_data: Dict[str, Any],
        tiles_batch_info: List[Dict[str, Any]],
        dataset_type: InferenceType,
    ) -> None:
        """
        Process and tile annotations based on the dataset type.

        Args:
            coco_data: Original COCO data.
            coco_image_info: Information about the current image being processed.
            output_coco_data: Dict to store the processed COCO data.
            tiles_batch_info: List of dictionaries containing information about each tile.
            dataset_type: Type of the dataset (OBJECT_DETECTION, SEGMENTATION, or CLASSIFICATION).
        """
        annotations = [
            annotation
            for annotation in coco_data["annotations"]
            if annotation["image_id"] == coco_image_info["id"]
        ]

        if dataset_type in [
            InferenceType.OBJECT_DETECTION,
            InferenceType.SEGMENTATION,
        ]:
            self._process_object_detection_and_segmentation_annotations(
                annotations=annotations,
                tiles_batch_info=tiles_batch_info,
                output_coco_data=output_coco_data,
                dataset_type=dataset_type,
            )
        elif dataset_type == InferenceType.CLASSIFICATION:
            if len(annotations) == 1:
                self._process_classification_image(
                    annotation=annotations[0],
                    tiles_batch_info=tiles_batch_info,
                    output_coco_data=output_coco_data,
                )

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
