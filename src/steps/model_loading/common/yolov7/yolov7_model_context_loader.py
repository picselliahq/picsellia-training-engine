import os.path

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.ultralytics.ultralytics_model_context import (
    UltralyticsModelContext,
)
from src.models.model.yolov7_model_context import Yolov7ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_loading.common.ultralytics.ultralytics_model_context_loader import (
    ultralytics_load_model,
)


@step
def yolov7_model_context_loader(
    model_context: Yolov7ModelContext, weights_path_to_load: str
) -> Yolov7ModelContext:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        loaded_model = ultralytics_load_model(
            weights_path_to_load=weights_path_to_load,
            device=context.hyperparameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model_context
