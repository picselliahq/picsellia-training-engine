from picsellia.types.enums import InferenceType

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.ultralytics.ultralytics_model_context import (
    UltralyticsModelContext,
)
from src.models.steps.model_evaluation.model_evaluator import ModelEvaluator
from src.models.steps.model_inferencing.ultralytics.classification_model_context_inference import (
    UltralyticsClassificationModelContextInference,
)


@step
def ultralytics_model_evaluator(
    model_context: UltralyticsModelContext,
    dataset_context: TDatasetContext,
):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    if dataset_context.dataset_version.type == InferenceType.CLASSIFICATION:
        model_inference = UltralyticsClassificationModelContextInference(
            model_context=model_context,
        )
    else:
        raise (ValueError("Inference type not supported"))

    model_evaluator = ModelEvaluator(
        model_inference=model_inference,
        dataset_context=dataset_context,
        experiment=context.experiment,
    )

    model_evaluator.evaluate()
