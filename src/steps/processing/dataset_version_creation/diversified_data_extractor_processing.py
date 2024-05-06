from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.dataset_context import DatasetContext
from src.models.parameters.processing.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from src.models.steps.processing.dataset_version_creation.diversified_data_extractor_processing import (
    DiversifiedDataExtractorProcessing,
)
from src.steps.model_loader.processing.processing_diversified_data_extractor_model_loader import (
    EmbeddingModel,
)


@step
def diversified_data_extractor_processing(
    dataset_context: DatasetContext, embedding_model: EmbeddingModel
):
    context: PicselliaProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    processor = DiversifiedDataExtractorProcessing(
        client=context.client,
        input_dataset_context=dataset_context,
        output_dataset_version=context.output_dataset_version,
        embedding_model=embedding_model,
        distance_threshold=context.processing_parameters.distance_threshold,
    )
    processor.process()
