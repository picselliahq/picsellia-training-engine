import os

from PIL import Image
from picsellia.types.enums import InferenceType
from picsellia_annotations.coco import Annotation

from src.models.dataset.common.dataset_collection import DatasetCollection


class BoundingBoxCropperProcessing:
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.

    It processes the images in the input dataset version to extract the bounding boxes for the specified label.
    The extracted images are saved to the output dataset version.

    Attributes:
        dataset_collection (DatasetCollection): The dataset collection containing the input and output dataset versions.
        label_name_to_extract (str): The name of the label to extract the bounding boxes for.
    """

    def __init__(
        self,
        dataset_collection: DatasetCollection,
        label_name_to_extract: str,
    ):
        self.dataset_collection = dataset_collection
        self.label_name_to_extract = label_name_to_extract

    def _update_output_dataset_version(self):
        """
        Updates the output dataset version with the description and type.
        """
        output_dataset_type = InferenceType.CLASSIFICATION
        input_dataset_context = self.dataset_collection["input"]
        output_dataset_description = (
            f"Dataset extracted from dataset version "
            f"'{input_dataset_context.dataset_version.version}' "
            f"(id: {input_dataset_context.dataset_version.id}) in dataset "
            f"'{input_dataset_context.dataset_version.name}' with label '{self.label_name_to_extract}'."
        )
        self.dataset_collection["output"].dataset_version.update(
            description=output_dataset_description, type=output_dataset_type
        )

    def _process_dataset_collection(self):
        """
        Processes the images in the input dataset version to extract the bounding boxes for the specified label.
        """
        for image_filename in os.listdir(self.dataset_collection["input"].images_dir):
            self._process_image(image_filename)

    def _process_image(self, image_filename: str) -> None:
        """
        Processes an image to extract the bounding box for the specified label.
        If the label is found in the image's annotations, the bounding box is extracted and saved to the processed dataset directory.
        Args:
            image_filename (str): The filename of the image to process.
        """
        image_filepath = os.path.join(
            self.dataset_collection["input"].images_dir, image_filename
        )
        image = Image.open(image_filepath)
        coco_files_image_ids = [
            coco_file_image.id
            for coco_file_image in self.dataset_collection["input"].coco_file.images
            if coco_file_image.file_name == image_filename
        ]
        if coco_files_image_ids:
            image_id_coco_file = coco_files_image_ids[0]
            coco_file_annotations = [
                coco_file_annotation
                for coco_file_annotation in self.dataset_collection[
                    "input"
                ].coco_file.annotations
                if coco_file_annotation.image_id == image_id_coco_file
            ]
            for coco_file_annotation in coco_file_annotations:
                label = [
                    category.name
                    for category in self.dataset_collection[
                        "input"
                    ].coco_file.categories
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
            self.dataset_collection["output"].images_dir, self.label_name_to_extract
        )
        os.makedirs(label_folder, exist_ok=True)

        processed_image_filename = f"{os.path.splitext(image_filename)[0]}_{self.label_name_to_extract}_{coco_file_annotation.id}.{image_filename.split('.')[-1]}"
        processed_image_filepath = os.path.join(label_folder, processed_image_filename)

        extracted_image.save(processed_image_filepath)

    def process(self) -> DatasetCollection:
        """
        Processes the images in the dataset version to extract the bounding boxes for the specified label and adds them to the output dataset version.
        """
        self._update_output_dataset_version()
        self._process_dataset_collection()
        return self.dataset_collection
