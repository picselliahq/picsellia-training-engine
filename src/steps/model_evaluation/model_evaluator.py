from typing import Union

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.steps.model_evaluation.model_evaluator import ModelEvaluator
from src.models.steps.model_inferencing.model_collection_inference import (
    ModelCollectionInference,
)
from src.models.steps.model_inferencing.model_context_inference import (
    ModelContextInference,
)


@step
def model_evaluator(
    model_inference: Union[ModelContextInference, ModelCollectionInference],
    dataset_context: TDatasetContext,
):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_evaluator = ModelEvaluator(
        model_inference=model_inference,
        dataset_context=dataset_context,
        experiment=context.experiment,
    )

    model_evaluator.evaluate()
