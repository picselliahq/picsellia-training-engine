import os
from typing import List, Tuple

from picsellia import DatasetVersion, Client
from picsellia.types.enums import AnnotationFileType
from sahi.slicing import slice_coco

from src.models.dataset.dataset_context import DatasetContext
from src.models.steps.processing.dataset_version_creation.dataset_version_creation_processing import (
    DatasetVersionCreationProcessing,
)

from sahi.utils.coco import CocoAnnotation
from shapely.geometry import box, MultiPolygon, GeometryCollection, Polygon
from shapely.validation import make_valid


def new_get_sliced_coco_annotation(self, slice_bbox: List[int]):
    shapely_polygon = box(slice_bbox[0], slice_bbox[1], slice_bbox[2], slice_bbox[3])
    samp = self._shapely_annotation.multipolygon
    if not samp.is_valid:
        valid = make_valid(samp)
        if not isinstance(valid, MultiPolygon):
            if isinstance(valid, GeometryCollection):
                polygons = [
                    geom
                    for geom in valid.geoms
                    if isinstance(geom, (Polygon, MultiPolygon))
                ]
                if polygons:
                    valid = MultiPolygon(polygons)
                else:
                    raise ValueError(
                        "Aucun polygone valide trouvé dans la GeometryCollection"
                    )
            else:
                valid = MultiPolygon([valid])
        self._shapely_annotation.multipolygon = valid
    intersection_shapely_annotation = self._shapely_annotation.get_intersection(
        shapely_polygon
    )
    return CocoAnnotation.from_shapely_annotation(
        intersection_shapely_annotation,
        category_id=self.category_id,
        category_name=self.category_name,
        iscrowd=self.iscrowd,
    )


class SlicerProcessing(DatasetVersionCreationProcessing):
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.
    """

    def __init__(
        self,
        client: Client,
        input_dataset_context: DatasetContext,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        min_area_ratio: float,
        output_dataset_version: DatasetVersion,
        datalake: str,
        destination_path: str,
    ):
        super().__init__(
            client=client,
            output_dataset_version=output_dataset_version,
            dataset_type=input_dataset_context.dataset_version.type,
            dataset_description=f"Dataset sliced from dataset version "
            f"'{input_dataset_context.dataset_version.version}' "
            f"(id: {input_dataset_context.dataset_version.id}) "
            f"in dataset '{input_dataset_context.dataset_version.name}' "
            f"with slice size {slice_height}x{slice_width}.",
            datalake=datalake,
        )
        self.dataset_context = input_dataset_context
        self.processed_dataset_context = DatasetContext(
            dataset_name="processed_dataset",
            dataset_version=self.output_dataset_version,
            destination_path=destination_path,
            multi_asset=None,
            labelmap=None,
            use_id=False,
        )
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.min_area_ratio = min_area_ratio

    def _process_images(self) -> Tuple[List[str], str]:
        """
        Processes all images in the dataset to extract the bounding boxes for the specified label.
        Returns:

        """
        coco_annotation_file_path = (
            self.dataset_context.dataset_version.export_annotation_file(
                annotation_file_type=AnnotationFileType.COCO,
                target_path=os.path.join(
                    self.dataset_context.destination_path, "annotations"
                ),
            )
        )
        CocoAnnotation.get_sliced_coco_annotation = new_get_sliced_coco_annotation
        slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=self.dataset_context.image_dir,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            min_area_ratio=self.min_area_ratio,
            output_coco_annotation_file_name="sliced_annotations",
            output_dir=self.processed_dataset_context.destination_path,
        )

        output_files = os.listdir(self.processed_dataset_context.destination_path)
        sliced_images_list = [
            os.path.join(self.processed_dataset_context.destination_path, file)
            for file in output_files
            if not file.endswith(".json")
        ]
        sliced_annotations_path = [
            os.path.join(self.processed_dataset_context.destination_path, file)
            for file in output_files
            if file.endswith(".json")
        ][0]
        return sliced_images_list, sliced_annotations_path

    def process(self) -> None:
        sliced_images_list, sliced_annotations_path = self._process_images()
        self._add_images_to_dataset_version(
            images_to_upload=sliced_images_list, images_tags=["picsellia_sliced"]
        )
        self._add_coco_annotations_to_dataset_version(
            annotation_path=sliced_annotations_path
        )
