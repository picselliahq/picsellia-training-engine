from src import pipeline
from src.models.contexts.picsellia_context import PicselliaProcessingContext

from src.models.parameters.processing.processing_easyorc_parameters import ProcessingEasyOcrParameters
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_data_extractor,
)

from src.steps.data_validation.processing.processing_easyocr_data_validator import easyocr_data_validator
from src.steps.model_loader.processing.processing_diversified_data_extractor_model_loader import (
    diversified_data_extractor_model_loader,
)
from src.steps.processing.dataset_version_creation.diversified_data_extractor_processing import (
    diversified_data_extractor_processing,
)
from src.steps.processing.pre_annotation.easyocr_processing import easyocr_processing
from src.steps.weights_validation.processing.processing_diversified_data_extractor_weights_validator import (
    diversified_data_extractor_weights_validator,
)


def get_context() -> (
        PicselliaProcessingContext[ProcessingEasyOcrParameters]
):
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingEasyOcrParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def easyocr_pipeline() -> None:
    dataset_context = processing_data_extractor(skip_asset_listing=True)

    easyocr_data_validator(dataset_context=dataset_context)

    easyocr_processing(
        dataset_context=dataset_context, embedding_model=embedding_model
    )


if __name__ == "__main__":
    easyocr_pipeline()
