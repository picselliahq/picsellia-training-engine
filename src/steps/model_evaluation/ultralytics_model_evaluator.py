from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.common.model_context import ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_evaluation.model_evaluator import ModelEvaluator
from src.models.steps.model_inferencing.ultralytics.classification_model_context_inference import (
    UltralyticsClassificationModelContextInference,
)


@step
def ultralytics_model_evaluator(
    model_context: ModelContext,
    dataset_context: TDatasetContext,
) -> None:
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    # 1. Preprocessing to apply on the test set (framework related -> dedicated code (in: dataset context | out: ))
    # 2. Batch preparation (batch_size) (optional)
    # 3. For batch in batches -> run model inference
    # 4. Postprocess all the inferences

    # inference_maker = UltralyticsInferenceMaker()
    # results: ClassificationResult = inference_maker.predict(data=test_set, batch_size=batch_size))

    # classification_metrics_computer = ClassificationMetricsComputer()
    # metrics: PicselliaMetricsResult = classification_metrics_computer.compute(results)

    # send_metrics_to_picsellia(metrics)

    # 5. Compute the metrics / confusion matrices (e.g. scikit classification report) on those inferences results
    # 6. Send to picsellia

    # model_inference = UltralyticsClassificationModelContextInference(
    #     model_context=model_context
    # )
    # predictions = model_inference._run_inference(image_paths)
    # prediction_classification_result = model_inference._post_process(
    #     image_paths, predictions, dataset_context
    # )

    model_inference = UltralyticsClassificationModelContextInference(
        model_context=model_context
    )
    image_paths = model_inference.preprocess_dataset(dataset_context=dataset_context)
    batches = model_inference.prepare_batches(
        image_paths=image_paths, batch_size=context.hyperparameters.batch_size
    )
    results = model_inference.run_inference_on_batches(batches=batches)
    classifications_predictions = model_inference.post_process_batches(
        batches=batches, results=results, dataset_context=dataset_context
    )
    picsellia_classifications_predictions = model_inference.get_picsellia_predictions(
        dataset_context=dataset_context, prediction_result=classifications_predictions
    )

    model_evaluator = ModelEvaluator(experiment=context.experiment)
    model_evaluator.evaluate(
        picsellia_predictions=picsellia_classifications_predictions
    )
