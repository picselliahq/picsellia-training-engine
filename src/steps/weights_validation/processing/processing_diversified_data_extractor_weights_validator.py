from src import step, Pipeline
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.parameters.processing.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from src.steps.model_loader.processing.processing_diversified_data_extractor_model_loader import (
    is_embedding_model_name_valid,
    SupportedEmbeddingModels,
)

import open_clip


@step
def diversified_data_extractor_weights_validator() -> str:
    context: PicselliaProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    pretrained_weights = context.processing_parameters.pretrained_weights
    model_architecture = context.processing_parameters.model_architecture

    available_pretrained_weights = open_clip.pretrained.list_pretrained_tags_by_model(
        model=model_architecture
    )

    embedding_model_name = context.processing_parameters.embedding_model

    if is_embedding_model_name_valid(
        source=embedding_model_name, target=SupportedEmbeddingModels.OPENCLIP
    ):
        if pretrained_weights not in available_pretrained_weights:
            raise ValueError(
                f"The provided pretrained weights '{context.processing_parameters.pretrained_weights}' are not available. "
                f"Available pretrained weights are {open_clip.pretrained.list_pretrained_tags_by_model(model_architecture)}."
            )
        return pretrained_weights

    else:
        raise ValueError(
            f"The provided model '{context.processing_parameters.embedding_model}' is not supported yet. "
            f"Supported models are {[member.name.lower() for member in SupportedEmbeddingModels]}."
        )
