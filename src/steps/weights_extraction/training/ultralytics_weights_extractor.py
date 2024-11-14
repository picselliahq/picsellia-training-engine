import os
from typing import Optional

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.ultralytics.ultralytics_model_context import (
    UltralyticsModelContext,
)


@step
def ultralytics_model_context_extractor(
    pretrained_weights_name: Optional[str] = None,
    trained_weights_name: Optional[str] = None,
    config_name: Optional[str] = None,
    exported_weights_name: Optional[str] = None,
) -> UltralyticsModelContext:
    """
    Extracts a model context from the active Picsellia training experiment.

    This function retrieves the active training context from the pipeline and extracts the base model version
    from the experiment. It then creates a `ModelContext` object for the model, specifying the name and pretrained
    weights. The function downloads the necessary model weights to a specified directory and returns the
    initialized `ModelContext`.

    Returns:
        ModelContext: The extracted and initialized model context with the downloaded weights.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_version = context.experiment.get_base_model_version()
    model_context = UltralyticsModelContext(
        model_name=model_version.name,
        model_version=model_version,
        pretrained_weights_name=pretrained_weights_name,
        trained_weights_name=trained_weights_name,
        config_name=config_name,
        exported_weights_name=exported_weights_name,
    )
    model_context.download_weights(
        destination_path=os.path.join(os.getcwd(), context.experiment.name, "model")
    )
    return model_context
