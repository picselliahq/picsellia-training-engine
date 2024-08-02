# type: ignore
import dataclasses
from argparse import ArgumentParser

from picsellia.types.enums import ProcessingType

from src import pipeline
from src.models.contexts.processing.test_picsellia_processing_context import (
    TestPicselliaProcessingContext,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_dataset_collection_extractor,
)
from src.steps.data_validation.processing.processing_tiler_data_validator import (
    tiler_data_validator,
)
from src.steps.processing.dataset_version_creation.dataset_context_uploader import (
    dataset_context_uploader,
)
from src.steps.processing.dataset_version_creation.tiler_processing import (
    tiler_processing,
)


@dataclasses.dataclass
class ProcessingTilerParameters:
    tile_height: int = 512
    tile_width: int = 512
    overlap_height_ratio: float = 0.1
    overlap_width_ratio: float = 0.1
    min_area_ratio: float = 0.1
    datalake: str = "default"
    data_tag: str = None


parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--job_id", type=str)
parser.add_argument("--input_dataset_version_id", type=str)
parser.add_argument("--output_dataset_version_id", type=str)
parser.add_argument("--tile_height", type=int, default=512)
parser.add_argument("--tile_width", type=int, default=512)
parser.add_argument("--overlap_height_ratio", type=float, default=0.1)
parser.add_argument("--overlap_width_ratio", type=float, default=0.1)
parser.add_argument("--min_area_ratio", type=float, default=0.1)
parser.add_argument("--datalake", type=str, default="default")
parser.add_argument("--data_tag", type=str)

args = parser.parse_args()


def get_context() -> TestPicselliaProcessingContext:
    return TestPicselliaProcessingContext(
        api_token=args.api_token,
        organization_id=args.organization_id,
        job_id=args.job_id,
        job_type=ProcessingType.DATASET_VERSION_CREATION,
        input_dataset_version_id=args.input_dataset_version_id,
        output_dataset_version_id=args.output_dataset_version_id,
        processing_parameters=ProcessingTilerParameters(
            tile_height=args.tile_height,
            tile_width=args.tile_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            min_area_ratio=args.min_area_ratio,
            datalake=args.datalake,
            data_tag=args.data_tag,
        ),
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def tiler_processing_pipeline() -> None:
    dataset_collection = processing_dataset_collection_extractor()
    tiler_data_validator(dataset_context=dataset_collection.input)
    output_dataset_context = tiler_processing(dataset_collection=dataset_collection)
    dataset_context_uploader(dataset_context=output_dataset_context)


if __name__ == "__main__":
    tiler_processing_pipeline()
