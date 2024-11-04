from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.yolov7_model_context import Yolov7ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.yolov7.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from src.models.parameters.training.yolov7.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)


@step
def yolov7_model_context_preparator(
    model_context: Yolov7ModelContext,
) -> Yolov7ModelContext:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    input_hyperparameters = {
        "lr0": context.hyperparameters.lr0,
        "lrf": context.hyperparameters.lrf,
        "momentum": context.hyperparameters.momentum,
        "weight_decay": context.hyperparameters.weight_decay,
        "warmup_epochs": context.hyperparameters.warmup_epochs,
        "warmup_momentum": context.hyperparameters.warmup_momentum,
        "warmup_bias_lr": context.hyperparameters.warmup_bias_lr,
        "box": context.hyperparameters.box,
        "cls": context.hyperparameters.cls,
        "cls_pw": context.hyperparameters.cls_pw,
        "obj": context.hyperparameters.obj,
        "obj_pw": context.hyperparameters.obj_pw,
        "iou_t": context.hyperparameters.iou_t,
        "anchor_t": context.hyperparameters.anchor_t,
        "fl_gamma": context.hyperparameters.fl_gamma,
        "hsv_h": context.augmentation_parameters.hsv_h,
        "hsv_s": context.augmentation_parameters.hsv_s,
        "hsv_v": context.augmentation_parameters.hsv_v,
        "degrees": context.augmentation_parameters.degrees,
        "translate": context.augmentation_parameters.translate,
        "scale": context.augmentation_parameters.scale,
        "shear": context.augmentation_parameters.shear,
        "perspective": context.augmentation_parameters.perspective,
        "flipud": context.augmentation_parameters.flipud,
        "fliplr": context.augmentation_parameters.fliplr,
        "mosaic": context.augmentation_parameters.mosaic,
        "mixup": context.augmentation_parameters.mixup,
        "copy_paste": context.augmentation_parameters.copy_paste,
        "paste_in": context.augmentation_parameters.paste_in,
        "loss_ota": context.hyperparameters.loss_ota,
    }

    if not model_context.hyperparameters_path:
        raise (ValueError("Hyperparameters path is not set"))

    model_context.update_hyperparameters(
        hyperparameters=input_hyperparameters,
        hyperparameters_path=model_context.hyperparameters_path,
    )

    return model_context
