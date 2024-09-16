# type: ignore

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.paddle_ocr.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.paddle_ocr.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from src.models.parameters.training.paddle_ocr.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from src.models.steps.model_evaluation.model_evaluator import ModelEvaluator
from src.models.steps.model_prediction.paddle_ocr_model_collection_predictor import (
    PaddleOCRModelCollectionPredictor,
)


@step
def paddle_ocr_model_collection_evaluator(
    model_collection: PaddleOCRModelCollection,
    dataset_context: TDatasetContext,
) -> None:
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_collection_predictor = PaddleOCRModelCollectionPredictor(
        model_collection=model_collection,
    )
    image_paths = model_collection_predictor.pre_process_dataset_context(
        dataset_context=dataset_context
    )
    image_batches = model_collection_predictor.prepare_batches(
        image_paths=image_paths,
        batch_size=min(
            context.hyperparameters.bbox_batch_size,
            context.hyperparameters.text_batch_size,
        ),
    )
    batch_results = model_collection_predictor.run_inference_on_batches(
        image_batches=image_batches
    )
    picsellia_ocr_predictions = model_collection_predictor.post_process_batches(
        image_batches=image_batches,
        batch_results=batch_results,
        dataset_context=dataset_context,
    )

    model_evaluator = ModelEvaluator(experiment=context.experiment)
    model_evaluator.evaluate(picsellia_predictions=picsellia_ocr_predictions)
