from src import Pipeline, step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.model.model_context import ModelContext
from src.models.steps.model_loading.yolox.yolox_model_context_loader import (
    load_yolox_weights,
)


@step
def yolox_model_context_loader(model_context: ModelContext) -> ModelContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    if (
        True
        # model_context.pretrained_model_path
        # and os.path.exists(model_context.pretrained_model_path)
    ):
        model_version_parameters = model_context.model_version.sync()
        architecture = model_version_parameters["docker_env_variables"]["architecture"]

        loaded_model = load_yolox_weights(
            model_path=f"{model_context.weights_dir}/yolox_s.pth",
            model_architecture=architecture,
            labelmap=[
                1,
            ]
            * 80,  # model_context.labelmap, TODO remove
            device=context.processing_parameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {model_context.pretrained_model_path}. Cannot load model."
        )
    return model_context
