import logging
import math
from typing import List, Optional, Set

from PIL import Image, ImageOps
import requests
import numpy as np
from picsellia import DatasetVersion, Client, Data
from scipy.spatial import KDTree

from src.models.dataset.dataset_context import DatasetContext
from src.models.steps.processing.dataset_version_creation.dataset_version_creation_processing import (
    DatasetVersionCreationProcessing,
)
from src.steps.model_loader.processing.processing_diversified_data_extractor_model_loader import (
    EmbeddingModel,
)
from tqdm import tqdm
from imagehash import phash

logger = logging.getLogger("picsellia")


class DiversifiedDataExtractorProcessing(DatasetVersionCreationProcessing):
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.
    """

    def __init__(
        self,
        client: Client,
        input_dataset_context: DatasetContext,
        output_dataset_version: DatasetVersion,
        embedding_model: EmbeddingModel,
        distance_threshold: float,
    ):
        super().__init__(
            client=client,
            output_dataset_version=output_dataset_version,
        )
        self.input_dataset_context = input_dataset_context
        self.embedding_model = embedding_model
        self.distance_threshold = distance_threshold

        self.skipped_similar_asset_number = 0
        self.skipped_identical_asset_number = 0
        self.skipped_error_asset_number = 0

    @property
    def input_dataset_version_size(self) -> int:
        return 34  # self.input_dataset_context.dataset_version.sync()["size"]

    @property
    def skipped_asset_number(self) -> int:
        return (
            self.skipped_similar_asset_number
            + self.skipped_identical_asset_number
            + self.skipped_error_asset_number
        )

    def fetch_and_prepare_image(self, image_url: str) -> Image:
        """Fetches and prepares an image for processing."""
        try:
            with requests.get(image_url, stream=True) as response:
                response.raise_for_status()
                image = Image.open(response.raw)
                if image.getexif().get(0x0112, 1) != 1:
                    image = ImageOps.exif_transpose(image)
                return image.convert("RGB") if image.mode != "RGB" else image
        except requests.RequestException as e:
            print(f"Failed to fetch or process image from {image_url}: {str(e)}")
            return None

    def compute_image_tensor(self, image: Image) -> np.ndarray:
        """Converts an image to a tensor."""
        preprocessed_image = self.embedding_model.apply_preprocessing(image)
        return self.embedding_model.encode_image(image=preprocessed_image)

    def process(self) -> None:
        perceptual_hash_set: Set[str] = set()
        tensor_list: List[np.ndarray] = []
        kd_tree: Optional[KDTree] = None

        input_dataset_version_size = self.input_dataset_version_size

        batch_size = 10
        current_offset = 0

        while current_offset < input_dataset_version_size:
            logger.info(
                f"Starting batch {int(current_offset/batch_size + 1)}"
                f"/{math.ceil(input_dataset_version_size/batch_size)}:"
            )
            try:
                assets_batch = self.input_dataset_context.get_assets_batch(
                    limit=batch_size, offset=current_offset
                )
            except Exception as e:
                logger.warning(f"Failed to fetch batch: {e}. Retrying...")
                assets_batch = self.input_dataset_context.get_assets_batch(
                    limit=batch_size, offset=current_offset
                )

            batch_to_upload: List[Data] = []

            for asset in tqdm(assets_batch, desc="Processing Images"):
                try:
                    image = self.fetch_and_prepare_image(asset.url)

                    if not image:
                        self.skipped_error_asset_number += 1
                        continue

                    img_hash = str(phash(image))

                    if img_hash in perceptual_hash_set:
                        self.skipped_identical_asset_number += 1
                        continue

                    perceptual_hash_set.add(img_hash)
                    tensor = self.compute_image_tensor(image=image)

                    if kd_tree is None:
                        kd_tree = KDTree(tensor)
                        tensor_list.append(tensor)
                        batch_to_upload.append(asset.get_data())
                    else:
                        nearest_neighbour_distance, _ = kd_tree.query(tensor)

                        if nearest_neighbour_distance > self.distance_threshold:
                            kd_tree = KDTree(np.vstack([kd_tree.data, tensor]))
                            tensor_list.append(tensor)
                            batch_to_upload.append(asset.get_data())
                        else:
                            self.skipped_similar_asset_number += 1

                except Exception as e:
                    self.skipped_error_asset_number += 1
                    logger.warning(f"Skipped asset due to exception: {e}")

            self._add_data_to_dataset_version(data=batch_to_upload)
            current_offset += batch_size

            logger.info(
                f"Uploaded {current_offset - self.skipped_asset_number} | "
                f"Skipped (errors) {self.skipped_error_asset_number} | "
                f"Skipped (identical) {self.skipped_identical_asset_number} | "
                f"Skipped (too similar) {self.skipped_similar_asset_number}"
            )
