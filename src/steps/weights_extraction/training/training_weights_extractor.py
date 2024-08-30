import os

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext


@step
def training_weights_extractor() -> ModelContext:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_version = context.experiment.get_base_model_version()
    model_context = ModelContext(
        model_name=model_version.name,
        model_version=model_version,
        destination_path=os.path.join(os.getcwd(), context.experiment.name),
        pretrained_model_filename="weights",
        config_filename="config",
        prefix_model_name=None,
    )
    model_context.download_weights(
        model_weights_destination_path=model_context.weights_dir
    )
    return model_context
