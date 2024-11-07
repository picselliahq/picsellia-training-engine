from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.dataset_context import TDatasetContext
from src.models.model.yolov7_model_context import Yolov7ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.yolov7.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from src.models.parameters.training.yolov7.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)
from src.models.steps.model_evaluation.common.model_evaluator import ModelEvaluator
from src.models.steps.model_prediction.common.yolov7.segmentation_model_context_predictor import (
    Yolov7SegmentationModelContextPredictor,
)


@step
def yolov7_model_context_evaluator(
    model_context: Yolov7ModelContext,
    dataset_context: TDatasetContext,
) -> None:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_predictor = Yolov7SegmentationModelContextPredictor(
        model_context=model_context
    )
    image_paths = model_context_predictor.pre_process_dataset_context(
        dataset_context=dataset_context
    )
    label_path_to_mask_paths = model_context_predictor.run_inference(
        image_paths=image_paths,
        hyperparameters=context.hyperparameters,
    )
    picsellia_polygons_predictions = model_context_predictor.post_process(
        label_path_to_mask_paths=label_path_to_mask_paths,
        dataset_context=dataset_context,
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment, inference_type=model_context.model_version.type
    )
    model_evaluator.evaluate(picsellia_predictions=picsellia_polygons_predictions)
