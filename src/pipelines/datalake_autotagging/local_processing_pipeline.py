# type: ignore
import dataclasses
from argparse import ArgumentParser

from src import pipeline
from src.models.contexts.processing.test_picsellia_datalake_processing_context import \
    TestPicselliaDatalakeProcessingContext
from src.pipelines.partner_model.mmdetection.configs.fpg.retinanet_r50_fpg_crop640_50e_coco import model
from src.steps.data_extraction.processing.processing_data_extractor import processing_datalake_extractor
from src.steps.model_loading.common.minigpt.minigpt_model_context_loader import minigpt_model_context_loader
from src.steps.processing.autotagging.datalake_autotagging import datalake_autotagging_processing
from src.steps.weights_extraction.processing.processing_weights_extractor import processing_model_context_extractor


@dataclasses.dataclass
class ProcessingDatalakeAutotaggingParameters:
    tags_list: list[str]


parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--job_id", type=str)
parser.add_argument("--input_datalake_id", type=str)
parser.add_argument("--output_datalake_id", type=str)
parser.add_argument("--model_version_id", type=str)
parser.add_argument("--tags_list", nargs="+", type=str)
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--limit", type=int, default=100)
args = parser.parse_args()


def get_context() -> TestPicselliaDatalakeProcessingContext:
    print(f'limit: {args.limit}')
    return TestPicselliaDatalakeProcessingContext(
        api_token=args.api_token,
        organization_id=args.organization_id,
        job_id=args.job_id,
        job_type=None,
        input_datalake_id=args.input_datalake_id,
        output_datalake_id=args.output_datalake_id,
        model_version_id=args.model_version_id,
        offset=args.offset,
        limit=args.limit,
        use_id=True,
        processing_parameters=ProcessingDatalakeAutotaggingParameters(
            tags_list=args.tags_list
        ),
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def datalake_autotagging_processing_pipeline() -> None:
    datalake = processing_datalake_extractor()
    model_context = processing_model_context_extractor()
    model_context = minigpt_model_context_loader(model_context=model_context)
    datalake_autotagging_processing(
        datalake=datalake
    )


if __name__ == "__main__":
    datalake_autotagging_processing_pipeline()
