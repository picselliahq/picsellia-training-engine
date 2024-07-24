from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from src.models.steps.model_evaluation.model_evaluator import ModelEvaluator
from src.models.steps.model_inferencing.paddle_ocr_model_collection_inference import (
    PaddleOCRModelCollectionInference,
)


@step
def paddle_ocr_model_evaluator(
    model_collection: PaddleOCRModelCollection,
    dataset_context: TDatasetContext,
):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_inference = PaddleOCRModelCollectionInference(
        model_collection=model_collection
    )

    model_evaluator = ModelEvaluator(
        model_inference=model_inference,
        dataset_context=dataset_context,
        experiment=context.experiment,
    )

    model_evaluator.evaluate()
