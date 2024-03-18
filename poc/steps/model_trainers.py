from ultralytics import YOLO

from poc.models.contexts.picsellia_context import PicselliaTrainingContext
from poc.pipeline import Pipeline
from poc.step import step


@step
def model_trainer(model: YOLO, callbacks: dict, dataset_path: str):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    # trainer_args = dict(
    #     project=context.experiment.name,
    #     data=dataset_path,
    #     **context["training_args"],
    #     **context["augmentation_args"],
    # )
    for callback_name, callback_function in callbacks.items():
        model.add_callback(event=callback_name, func=callback_function)

    model.train(
        data=dataset_path,
        project=context.experiment.name,
        seed=context.hyperparameters.seed,
        epochs=context.hyperparameters.epochs,
        batch=context.hyperparameters.batch_size,
        imgsz=context.hyperparameters.image_size,
        device=context.hyperparameters.device,
        cache=context.hyperparameters.use_cache,
        save_period=context.hyperparameters.save_period,
        val=context.hyperparameters.validate,
    )
    return model
