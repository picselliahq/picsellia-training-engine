import os

import cv2
import numpy as np
from picsellia import DatasetVersion, Client
from picsellia.types.enums import TagTarget, InferenceType
from picsellia_annotations.coco import Annotation

from src import Pipeline
from src import step
from src.models.dataset.dataset_context import DatasetContext
from src.steps.processing.utils.dataset_version_creation_processing import (
    DatasetVersionCreationProcessing,
)


class BoundingBoxCropperProcessing(DatasetVersionCreationProcessing):
    def __init__(
        self,
        client: Client,
        input_dataset_context: DatasetContext,
        label_name_to_extract: str,
        output_dataset_version: DatasetVersion,
        datalake_name: str,
        destination_path: str,
    ):
        super().__init__(
            client=client,
            output_dataset_version=output_dataset_version,
            dataset_type=InferenceType.CLASSIFICATION,
            dataset_description=f"Dataset extracted from dataset version "
            f"'{input_dataset_context.dataset_version.version}' "
            f"(id: {input_dataset_context.dataset_version.id}) "
            f"in dataset '{input_dataset_context.dataset_version.name}' "
            f"with label '{label_name_to_extract}'.",
            datalake_name=datalake_name,
        )
        self.dataset_context = input_dataset_context
        self.processed_dataset_context = DatasetContext(
            dataset_name="processed_dataset",
            dataset_version=self.output_dataset_version,
            destination_path=destination_path,
            multi_asset=None,
            labelmap=None,
        )
        self.label_name_to_extract = label_name_to_extract

    def _process_images(self) -> None:
        for image_filename in os.listdir(self.dataset_context.image_dir):
            self._process_image(image_filename)

    def _process_image(self, image_filename: str) -> None:
        image_filepath = os.path.join(self.dataset_context.image_dir, image_filename)
        image = cv2.imread(image_filepath)
        image_id_coco_files = [
            image_coco_file.id
            for image_coco_file in self.dataset_context.coco_file.images
            if image_coco_file.file_name == image_filename
        ]
        if image_id_coco_files:
            image_id_coco_file = image_id_coco_files[0]
            annotations = [
                annotation
                for annotation in self.dataset_context.coco_file.annotations
                if annotation.image_id == image_id_coco_file
            ]
            for annotation in annotations:
                label = [
                    category.name
                    for category in self.dataset_context.coco_file.categories
                    if category.id == annotation.category_id
                ][0]
                if label == self.label_name_to_extract:
                    self._extract(
                        image=image,
                        image_filename=image_filename,
                        annotation=annotation,
                    )

    def _extract(
        self, image: np.ndarray, image_filename: str, annotation: Annotation
    ) -> None:
        extracted_image = image[
            int(annotation.bbox[1]) : int(annotation.bbox[1]) + int(annotation.bbox[3]),
            int(annotation.bbox[0]) : int(annotation.bbox[0]) + int(annotation.bbox[2]),
        ]
        if extracted_image.shape[0] == 0 or extracted_image.shape[1] == 0:
            return

        label_folder = os.path.join(
            self.processed_dataset_context.image_dir, self.label_name_to_extract
        )
        os.makedirs(label_folder, exist_ok=True)

        processed_image_filename = f"{os.path.splitext(image_filename)[0]}_{self.label_name_to_extract}_{annotation.id}.{image_filename.split('.')[-1]}"
        processed_image_filepath = os.path.join(label_folder, processed_image_filename)

        cv2.imwrite(processed_image_filepath, extracted_image)

    def _add_processed_images_to_dataset_version(self) -> None:
        for label_folder in os.listdir(self.processed_dataset_context.image_dir):
            full_label_folder_path = os.path.join(
                self.processed_dataset_context.image_dir, label_folder
            )
            if os.path.isdir(full_label_folder_path):
                filepaths = [
                    os.path.join(full_label_folder_path, file)
                    for file in os.listdir(full_label_folder_path)
                ]
                self._add_images_to_dataset_version(
                    images_to_upload=filepaths,
                    images_tags=[label_folder, "processed_dataset"],
                )
        conversion_job = self.processed_dataset_context.dataset_version.convert_tags_to_classification(
            tag_type=TagTarget.ASSET,
            tags=self.processed_dataset_context.dataset_version.list_asset_tags(),
        )
        conversion_job.wait_for_done()

    def process(self) -> None:
        self._process_images()
        self._add_processed_images_to_dataset_version()


@step
def bounding_box_cropper_processing(dataset_context: DatasetContext):
    context = Pipeline.get_active_context()

    processor = BoundingBoxCropperProcessing(
        client=context.client,
        input_dataset_context=dataset_context,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
        output_dataset_version=context.output_dataset_version,
        datalake_name=context.processing_parameters.datalake_name,
        destination_path=os.path.join(os.getcwd(), str(context.job_id)),
    )
    processor.process()
