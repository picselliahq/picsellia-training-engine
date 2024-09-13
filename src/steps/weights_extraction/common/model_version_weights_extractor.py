import os

from src import Pipeline, step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.model.model_context import ModelContext


@step
def model_version_weights_extractor():
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_version = context.client.get_model_version_by_id(
        id=context.processing_parameters.model_version_id
    )
    model_context = ModelContext(
        model_name=model_version.name,
        model_version=model_version,
        destination_path=os.path.join(os.getcwd(), context.job_id),
        labelmap=model_version.labels,
    )

    model_context.download_weights(
        model_weights_destination_path=model_context.weights_dir
    )
    return model_context
