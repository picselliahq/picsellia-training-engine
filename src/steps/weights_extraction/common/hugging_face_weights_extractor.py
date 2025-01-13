import logging
from typing import Optional

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
    hugging_face_model_name: Optional[str] = None,
) -> HuggingFaceModelContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    model_version = context.model_version
    if not hugging_face_model_name:
        model_parameters = model_version.sync()["parameters"]
        hugging_face_model_name = model_parameters.get("hugging_face_model_name")
        if not hugging_face_model_name:
            raise ValueError(
                "Hugging Face model name not provided. Please provide it as an argument or set the 'hugging_face_model_name' parameter in the model version."
            )
    print(f"Loading Hugging Face model {hugging_face_model_name}")
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
