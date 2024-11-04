import logging

from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.model.huggingface.hugging_face_model_context import (
    HuggingFaceModelContext,
)

logger = logging.getLogger(__name__)


@step
def hugging_face_model_context_extractor(
    hugging_face_model_name: str,
) -> HuggingFaceModelContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    model_version = context.model_version
    model_context = HuggingFaceModelContext(
        hugging_face_model_name=hugging_face_model_name,
        model_name=model_version.name,
        model_version=model_version,
        pretrained_weights_name=None,
        trained_weights_name=None,
        config_name=None,
        exported_weights_name=None,
    )
    return model_context
