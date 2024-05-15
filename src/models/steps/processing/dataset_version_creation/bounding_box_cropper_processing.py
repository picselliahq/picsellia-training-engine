import os

from PIL import Image
from picsellia import DatasetVersion, Client, Datalake
from picsellia.types.enums import TagTarget, InferenceType
from picsellia_annotations.coco import Annotation

from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.processing.dataset_version_creation.dataset_version_creation_processing import (
    DatasetVersionCreationProcessing,
)


class BoundingBoxCropperProcessing(DatasetVersionCreationProcessing):
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.
    """

    def __init__(
        self,
        client: Client,
        datalake: Datalake,
        input_dataset_context: DatasetContext,
        label_name_to_extract: str,
        output_dataset_version: DatasetVersion,
        destination_path: str,
    ):
        super().__init__(
            client=client,
            output_dataset_version=output_dataset_version,
            output_dataset_type=InferenceType.CLASSIFICATION,
            output_dataset_description=f"Dataset extracted from dataset version "
            f"'{input_dataset_context.dataset_version.version}' "
            f"(id: {input_dataset_context.dataset_version.id}) "
            f"in dataset '{input_dataset_context.dataset_version.name}' "
            f"with label '{label_name_to_extract}'.",
            dataset_type=InferenceType.CLASSIFICATION,
            datalake=datalake,
            output_dataset_version=output_dataset_version,
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

    @property
    def output_dataset_description(self) -> str:
        """
        Returns the description of the output dataset version.

        Returns:
            str: The description of the output dataset version.
        """
        return (
            f"Dataset extracted from dataset version "
            f"'{self.dataset_context.dataset_version.version}' "
            f"(id: {self.dataset_context.dataset_version.id}) in dataset "
            f"'{self.dataset_context.dataset_version.name}' with label '{self.label_name_to_extract}'."
        )

    def _process_images(self) -> None:
        """
        Processes all images in the dataset to extract the bounding boxes for the specified label.
        Returns:

        """
        for image_filename in os.listdir(self.dataset_context.image_dir):
            self._process_image(image_filename)

    def _process_image(self, image_filename: str) -> None:
        """
        Processes an image to extract the bounding box for the specified label.
        If the label is found in the image's annotations, the bounding box is extracted and saved to the processed dataset directory.
        Args:
            image_filename (str): The filename of the image to process.
        """
        image_filepath = os.path.join(self.dataset_context.image_dir, image_filename)
        image = Image.open(image_filepath)
        coco_files_image_ids = [
            coco_file_image.id
            for coco_file_image in self.dataset_context.coco_file.images
            if coco_file_image.file_name == image_filename
        ]
        if coco_files_image_ids:
            image_id_coco_file = coco_files_image_ids[0]
            coco_file_annotations = [
                coco_file_annotation
                for coco_file_annotation in self.dataset_context.coco_file.annotations
                if coco_file_annotation.image_id == image_id_coco_file
            ]
            for coco_file_annotation in coco_file_annotations:
                label = [
                    category.name
                    for category in self.dataset_context.coco_file.categories
                    if category.id == coco_file_annotation.category_id
                ][0]
                if label == self.label_name_to_extract:
                    self._extract(
                        image=image,
                        image_filename=image_filename,
                        coco_file_annotation=coco_file_annotation,
                    )

    def _extract(
        self, image: Image, image_filename: str, coco_file_annotation: Annotation
    ) -> None:
        """
        Extracts the bounding box from the image and saves it to the processed dataset directory.

        Args:
            image (Image): The image to extract the bounding box from.
            image_filename (str): The filename of the image.
            coco_file_annotation (Annotation): The annotation containing the bounding box.
        """
        x = int(coco_file_annotation.bbox[0])
        y = int(coco_file_annotation.bbox[1])
        width = int(coco_file_annotation.bbox[2])
        height = int(coco_file_annotation.bbox[3])

        extracted_image = image.crop((x, y, x + width, y + height))

        label_folder = os.path.join(
            self.processed_dataset_context.image_dir, self.label_name_to_extract
        )
        os.makedirs(label_folder, exist_ok=True)

        processed_image_filename = f"{os.path.splitext(image_filename)[0]}_{self.label_name_to_extract}_{coco_file_annotation.id}.{image_filename.split('.')[-1]}"
        processed_image_filepath = os.path.join(label_folder, processed_image_filename)

        extracted_image.save(processed_image_filepath)

    def _add_processed_images_to_dataset_version(self) -> None:
        """
        Adds the processed images to the dataset version.
        """
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
                    images_tags=[f"picsellia_tile_{label_folder}"],
                )
        conversion_job = self.processed_dataset_context.dataset_version.convert_tags_to_classification(
            tag_type=TagTarget.ASSET,
            tags=self.processed_dataset_context.dataset_version.list_asset_tags(),
        )
        conversion_job.wait_for_done()

    def process(self) -> None:
        """
        Processes the images in the dataset version to extract the bounding boxes for the specified label and adds them to the output dataset version.
        """
        self.update_output_dataset_version_description(
            description=self.output_dataset_description
        )
        self.update_output_dataset_version_inference_type(
            inference_type=InferenceType.CLASSIFICATION
        )
        self._process_images()
        self._add_processed_images_to_dataset_version()
