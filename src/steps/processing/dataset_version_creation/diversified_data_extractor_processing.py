from src import Pipeline, step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
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
        datalake=context.client.get_datalake(),
        input_dataset_context=dataset_context,
        output_dataset_version=context.output_dataset_version,
        embedding_model=embedding_model,
        distance_threshold=context.processing_parameters.distance_threshold,
    )
    processor.process()
