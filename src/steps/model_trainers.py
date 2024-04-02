from ultralytics import YOLO

from src.models.contexts.picsellia_context import PicselliaTrainingContext
from src import Pipeline
from src import step


@step
def model_trainer(model: YOLO, callbacks: dict, dataset_path: str):
    context: PicselliaTrainingContext = Pipeline.get_active_context()

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
        hsv_h=context.augmentation_parameters.hsv_h,
        hsv_s=context.augmentation_parameters.hsv_s,
        hsv_v=context.augmentation_parameters.hsv_v,
        degrees=context.augmentation_parameters.degrees,
        translate=context.augmentation_parameters.translate,
        scale=context.augmentation_parameters.scale,
        shear=context.augmentation_parameters.shear,
        perspective=context.augmentation_parameters.perspective,
        flipud=context.augmentation_parameters.flipud,
        fliplr=context.augmentation_parameters.fliplr,
        mosaic=context.augmentation_parameters.mosaic,
        mixup=context.augmentation_parameters.mixup,
        copy_paste=context.augmentation_parameters.copy_paste,
        auto_augment=context.augmentation_parameters.auto_augment,
        erasing=context.augmentation_parameters.erasing,
    )
    return model
