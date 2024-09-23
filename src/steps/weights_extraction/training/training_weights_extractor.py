import os

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext


@step
def training_model_context_extractor() -> ModelContext:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_version = context.experiment.get_base_model_version()
    model_context = ModelContext(
        model_name=model_version.name,
        model_version=model_version,
        pretrained_weights_name="pretrained-weights",
        trained_weights_name=None,
        config_name=None,
        exported_weights_name=None,
    )
    model_context.download_weights(
        destination_path=os.path.join(os.getcwd(), context.experiment.name, "model")
    )
    return model_context
