import os

import picsellia.exceptions

from src import Pipeline, step
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_loading.yolox_model_context_loader import load_yolox_weights


@step
def yolox_model_context_loader(model_context: ModelContext) -> ModelContext:
    context = Pipeline.get_active_context()

    if model_context.weights_dir and os.path.exists(model_context.weights_dir):
        model_version_parameters = model_context.model_version.sync()
        architecture = model_version_parameters["docker_env_variables"]["architecture"]

        try:
            model_file_name = context.model_version.get_file(
                name=context.processing_parameters.input_model_file_name
            ).filename

        except picsellia.exceptions.ResourceNotFoundError:
            raise FileNotFoundError(
                f"Model file {context.processing_parameters.input_model_file_name} not found in "
                f"model version {context.model_version_id}. Cannot load model. "
                f"Available files: {', '.join([file.name for file in context.model_version.list_files()])}"
            )

        model_path = os.path.join(model_context.weights_dir, model_file_name)

        loaded_model = load_yolox_weights(
            model_path=model_path,
            model_architecture=architecture,
            labelmap=model_context.labelmap,
            device=context.processing_parameters.device,
        )
        model_context.set_loaded_model(loaded_model)

    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {model_context.weights_dir}. Cannot load model."
        )

    return model_context
