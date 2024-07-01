from src import pipeline
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.parameters.processing.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_data_extractor,
)
from src.steps.data_validation.processing.processing_diversified_data_extractor_data_validator import (
    diversified_data_extractor_data_validator,
)
from src.steps.model_loader.processing.processing_diversified_data_extractor_model_loader import (
    diversified_data_extractor_model_loader,
)
from src.steps.processing.dataset_version_creation.diversified_data_extractor_processing import (
    diversified_data_extractor_processing,
)
from src.steps.weights_validation.processing.processing_diversified_data_extractor_weights_validator import (
    diversified_data_extractor_weights_validator,
)


def get_context() -> (
    PicselliaProcessingContext[ProcessingDiversifiedDataExtractorParameters]
):
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingDiversifiedDataExtractorParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def diversified_data_extractor_pipeline() -> None:
    dataset_context = processing_data_extractor(skip_asset_listing=True)

    diversified_data_extractor_data_validator(dataset_context=dataset_context)
    pretrained_weights = diversified_data_extractor_weights_validator()
    embedding_model = diversified_data_extractor_model_loader(
        pretrained_weights=pretrained_weights
    )
    diversified_data_extractor_processing(
        dataset_context=dataset_context, embedding_model=embedding_model
    )


if __name__ == "__main__":
    diversified_data_extractor_pipeline()
