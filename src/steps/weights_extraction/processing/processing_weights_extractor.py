import logging
import os

from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.model.common.model_context import ModelContext
from src.steps.data_extraction.processing.processing_data_extractor import (
    get_destination_path,
)

logger = logging.getLogger(__name__)


@step
def processing_model_context_extractor() -> ModelContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_version = context.model_version
    model_context = ModelContext(
        model_name=model_version.name,
        model_version=model_version,
        pretrained_weights_name="checkpoints",
        trained_weights_name=None,
        config_name="config",
        exported_weights_name=None,
    )
    model_context.download_weights(
        destination_path=os.path.join(
            get_destination_path(job_id=context.job_id), "model"
        )
    )
    return model_context
