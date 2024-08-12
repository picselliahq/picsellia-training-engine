from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.model.model_context import ModelContext
from src.models.steps.weights_extraction.processing.processing_model_context_extractor import (
    ProcessingModelContextExtractor,
)


@step
def processing_weights_extractor() -> ModelContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    model_context_extractor = ProcessingModelContextExtractor(
        model_version=context.model_version, job_id=context.job_id
    )

    model_context = model_context_extractor.get_model_context()
    model_context.download_weights()
    return model_context
