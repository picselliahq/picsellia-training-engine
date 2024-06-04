import concurrent.futures
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import sahi
from PIL import Image
from picsellia import DatasetVersion, Client
from sahi.slicing import (
    slice_coco,
    SliceImageResult,
    logger,
    get_slice_bboxes,
    process_coco_annotations,
    SlicedImage,
    MAX_WORKERS,
)
from sahi.utils.coco import CocoAnnotation, CocoImage
from sahi.utils.cv import (
    IMAGE_EXTENSIONS_LOSSLESS,
    IMAGE_EXTENSIONS_LOSSY,
    read_image_as_pil,
)
from shapely.geometry import box, MultiPolygon, GeometryCollection, Polygon
from shapely.validation import make_valid

from src.models.dataset.dataset_context import DatasetContext
from src.models.steps.processing.dataset_version_creation.dataset_version_creation_processing import (
    DatasetVersionCreationProcessing,
)


def custom_get_sliced_coco_annotation(self, slice_bbox: List[int]):
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


def custom_slice_image(
    image: Union[str, Image.Image],
    coco_annotation_list: Optional[List[CocoAnnotation]] = None,
    output_file_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    auto_slice_resolution: bool = True,
    min_area_ratio: float = 0.1,
    out_ext: Optional[str] = None,
    verbose: bool = False,
) -> SliceImageResult:
    """Slice a large image into smaller windows. If output_file_name is given export
    sliced images.

    Args:
        image (str or PIL.Image): File path of image or Pillow Image to be sliced.
        coco_annotation_list (List[CocoAnnotation], optional): List of CocoAnnotation objects.
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix for lossless image formats and png for lossy formats ('.jpg','.jpeg').
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
        num_total_invalid_segmentation: int
            Number of invalid segmentation annotations.
    """

    # define verboseprint
    verboselog = logger.info if verbose else lambda *a, **k: None

    def _export_single_slice(image: np.ndarray, output_dir: str, slice_file_name: str):
        image_pil = read_image_as_pil(image)
        slice_file_path = str(Path(output_dir) / slice_file_name)
        # export sliced image
        # quaility is removed due to discussions/973,981, pull/956
        # image_pil.save(slice_file_path, quality="keep")
        image_pil.save(slice_file_path)
        image_pil.close()  # to fix https://github.com/obss/sahi/issues/565
        verboselog("sliced image path: " + slice_file_path)

    # create outdir if not present
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read image
    image_pil = read_image_as_pil(image)
    verboselog("image.shape: " + str(image_pil.size))

    image_width, image_height = image_pil.size
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        auto_slice_resolution=auto_slice_resolution,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(
        original_image_size=[image_height, image_width], image_dir=output_dir
    )

    image_pil_arr = np.asarray(image_pil)
    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        image_pil_slice = image_pil_arr[tly:bry, tlx:brx]

        # set image file suffixes
        slice_suffixes = "_".join(map(str, slice_bbox))
        if out_ext:
            suffix = out_ext
        elif hasattr(image_pil, "filename"):
            suffix = Path(getattr(image_pil, "filename")).suffix
            if suffix in IMAGE_EXTENSIONS_LOSSY:
                suffix = ".png"
            elif suffix in IMAGE_EXTENSIONS_LOSSLESS:
                suffix = Path(image_pil.filename).suffix
        else:
            suffix = ".png"

        # set image file name and path
        slice_file_name = f"{output_file_name}_{slice_suffixes}{suffix}"

        # create coco image
        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]
        coco_image = CocoImage(
            file_name=slice_file_name, height=slice_height, width=slice_width
        )

        # append coco annotations (if present) to coco image
        if coco_annotation_list:
            coco_annotation_list = [
                coco_annotation
                for coco_annotation in coco_annotation_list
                if coco_annotation.json["area"] != 0
                and (
                    coco_annotation.json["bbox"]
                    or coco_annotation.json["segmentation"][0]
                )
            ]
            for sliced_coco_annotation in process_coco_annotations(
                coco_annotation_list, slice_bbox, min_area_ratio
            ):
                coco_image.add_annotation(sliced_coco_annotation)

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=image_pil_slice,
            coco_image=coco_image,
            starting_pixel=[slice_bbox[0], slice_bbox[1]],
        )
        sliced_image_result.add_sliced_image(sliced_image)

    # export slices if output directory is provided
    if output_file_name and output_dir:
        conc_exec = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        conc_exec.map(
            _export_single_slice,
            sliced_image_result.images,
            [output_dir] * len(sliced_image_result),
            sliced_image_result.filenames,
        )

    verboselog(
        "Num slices: "
        + str(n_ims)
        + " slice_height: "
        + str(slice_height)
        + " slice_width: "
        + str(slice_width)
    )

    return sliced_image_result


sahi.slicing.slice_image = custom_slice_image


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
        CocoAnnotation.get_sliced_coco_annotation = custom_get_sliced_coco_annotation
        slice_coco(
            coco_annotation_file_path=self.dataset_context.coco_file_path,
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
            images_to_upload=sliced_images_list,
            images_tags=["picsellia_sliced", self.output_dataset_version.version],
        )
        self._add_coco_annotations_to_dataset_version(
            annotation_path=sliced_annotations_path
        )
