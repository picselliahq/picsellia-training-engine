from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.dataset_context import TDatasetContext
from src.models.model.common.model_context import ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_evaluation.common.model_evaluator import ModelEvaluator
from src.models.steps.model_prediction.common.ultralytics.classification_model_context_predictor import (
    UltralyticsClassificationModelContextPredictor,
)


@step
def ultralytics_model_context_evaluator(
    model_context: ModelContext,
    dataset_context: TDatasetContext,
) -> None:
    """
    Evaluates an Ultralytics classification model on a given dataset.

    This function retrieves the active training context from the pipeline, performs inference using
    the provided Ultralytics classification model on the dataset, and evaluates the predictions. It processes
    the dataset in batches, runs inference, and then logs the evaluation results to the experiment.

    Args:
        model_context (ModelContext): The Ultralytics model context to be evaluated.
        dataset_context (TDatasetContext): The dataset context containing the data for evaluation.

    Returns:
        None: The function performs evaluation and logs the results to the experiment but does not return any value.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_predictor = UltralyticsClassificationModelContextPredictor(
        model_context=model_context
    )
    image_paths = model_context_predictor.pre_process_dataset_context(
        dataset_context=dataset_context
    )
    image_batches = model_context_predictor.prepare_batches(
        image_paths=image_paths, batch_size=context.hyperparameters.batch_size
    )
    batch_results = model_context_predictor.run_inference_on_batches(
        image_batches=image_batches
    )
    picsellia_classifications_predictions = (
        model_context_predictor.post_process_batches(
            image_batches=image_batches,
            batch_results=batch_results,
            dataset_context=dataset_context,
        )
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment, inference_type=model_context.model_version.type
    )
    model_evaluator.evaluate(
        picsellia_predictions=picsellia_classifications_predictions
    )
