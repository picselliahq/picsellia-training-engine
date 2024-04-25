# type: ignore

from src import pipeline
from src.models.contexts.picsellia_context import (
    PicselliaProcessingContext,
)
from src.pipelines.bounding_box_cropper.utils.bounding_box_cropper_data_validator import (
    bounding_box_cropper_data_validator,
)
from src.pipelines.bounding_box_cropper.utils.bounding_box_cropper_parameters import (
    BoundingBoxCropperParameters,
)
from src.pipelines.bounding_box_cropper.utils.bounding_box_cropper_processing import (
    bounding_box_cropper_processing,
)
from src.steps.data_extraction.data_extractor import processing_data_extractor


def get_context() -> PicselliaProcessingContext[BoundingBoxCropperParameters]:
    return PicselliaProcessingContext(
        processing_parameters_cls=BoundingBoxCropperParameters,
    )


# @dataclasses.dataclass
# class BoundingBoxCropperParameters:
#     label_name_to_extract: str
#     datalake: str
#
#
# def get_context() -> TestPicselliaProcessingContext:
#     return TestPicselliaProcessingContext(
#         job_id="job_id",
#         job_type=ProcessingType.DATASET_VERSION_CREATION,
#         input_dataset_version_id="018ef59f-8846-7683-a53d-702c9b06e390",
#         output_dataset_version_id="018f1508-a0fd-748b-93a1-5a3e87c4aeb6",
#         processing_parameters=BoundingBoxCropperParameters(label_name_to_extract="chaîne_isolateur_verre",
#                                                            datalake="default")
#     )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def bounding_box_cropper_processing_pipeline() -> None:
    dataset_context = processing_data_extractor()
    bounding_box_cropper_data_validator(dataset_context=dataset_context)
    bounding_box_cropper_processing(dataset_context=dataset_context)


if __name__ == "__main__":
    bounding_box_cropper_processing_pipeline()
