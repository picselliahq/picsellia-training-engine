import json
import logging
import os
from tqdm import tqdm
from PIL import Image

from picsellia.types.enums import LogType

from super_gradients.training import Trainer, dataloaders, models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)

from super_gradients.training.utils.callbacks import Callback, PhaseContext
from super_gradients.common.environment.ddp_utils import multi_process_safe
from super_gradients.training.utils.callbacks.base_callbacks import PhaseContext
import yolonas_utils

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"

logging.getLogger("picsellia").setLevel(logging.INFO)


# classback class to log training metrics to picsellia
class SaveTrainingMetrics(Callback):
    @multi_process_safe
    def on_train_loader_end(self, context: PhaseContext) -> None:
        experiment.log(
            name="training_loss_cls",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss_cls"]),
        )
        experiment.log(
            name="training_loss_iou",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss_iou"]),
        )
        experiment.log(
            name="training_loss_dfl",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss_dfl"]),
        )
        experiment.log(
            name="training_loss",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss"]),
        )

    @multi_process_safe
    def on_validation_loader_end(self, context: PhaseContext) -> None:
        experiment.log(
            name="validation_loss_cls",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss_cls"]),
        )
        experiment.log(
            name="validation_loss_iou",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss_iou"]),
        )
        experiment.log(
            name="validation_loss_dfl",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss_dfl"]),
        )
        experiment.log(
            name="validation_loss",
            type=LogType.LINE,
            data=float(context.metrics_dict["PPYoloELoss/loss"]),
        )


experiment = yolonas_utils.get_experiment()
cwd = os.getcwd()
path_dict = {}
path_dict = yolonas_utils.create_yolo_dataset(experiment, cwd)  # get the data
labelmap, label_names = yolonas_utils.get_labelmap(experiment=experiment)
# trainer params
EXPERIMENT_NAME = experiment.name
# path to save the checkpoints to
CHECKPOINT_DIR = os.path.join(cwd, experiment.name, "checkpoints")

# dataset params
DATA_DIR = os.path.join(cwd, experiment.base_dir)
# child dir of DATA_DIR where train images are
TRAIN_IMAGES_DIR = path_dict["train"]["dataset_dir"]
# child dir of DATA_DIR where validation images are
VAL_IMAGES_DIR = path_dict["val"]["dataset_dir"]
# child dir of DATA_DIR where test images are
TEST_IMAGES_DIR = path_dict["test"]["dataset_dir"]

TRAIN_LABELS_DIR = os.path.join(
    cwd, path_dict["train"]["yolo_annotation_path"]
)  # child dir of DATA_DIR where train labels are
VAL_LABELS_DIR = os.path.join(
    cwd, path_dict["val"]["yolo_annotation_path"]
)  # child dir of DATA_DIR where validation labels are
TEST_LABELS_DIR = os.path.join(
    cwd, path_dict["test"]["yolo_annotation_path"]
)  # child dir of DATA_DIR where test labels are

CLASSES = label_names
NUM_CLASSES = len(CLASSES)
parameters = experiment.get_log(name="parameters").data

batch_size = parameters.get("batch_size", 4)
initial_lr = parameters.get("initial_lr", 0.0001)
max_epochs = parameters.get("max_epochs", 5)

DATALOADER_PARAMS = {"batch_size": parameters["batch_size"], "num_workers": 2}
# model params
MODEL_NAME = "yolo_nas_s"  # choose from yolo_nas_s, yolo_nas_m, yolo_nas_l
PRETRAINED_WEIGHTS = "coco"  # only one option here: coco

# get the checkpoints from experiment to experiment.base_dir
experiment.download_artifacts(with_tree=False)

train_data = coco_detection_yolo_format_train(
    dataset_params={
        "data_dir": DATA_DIR,
        "images_dir": TRAIN_IMAGES_DIR,
        "labels_dir": TRAIN_LABELS_DIR,
        "classes": CLASSES,
    },
    dataloader_params=DATALOADER_PARAMS,
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        "data_dir": DATA_DIR,
        "images_dir": VAL_IMAGES_DIR,
        "labels_dir": VAL_LABELS_DIR,
        "classes": CLASSES,
    },
    dataloader_params=DATALOADER_PARAMS,
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        "data_dir": DATA_DIR,
        "images_dir": TEST_IMAGES_DIR,
        "labels_dir": TEST_LABELS_DIR,
        "classes": CLASSES,
    },
    dataloader_params=DATALOADER_PARAMS,
)

trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)

artifact_namepath = experiment.get_artifact(
    name="checkpoints_best"
).filename  # path to the pretrained weights

# get the weights locally and feed it to function
model = models.get(
    "yolo_nas_s",
    checkpoint_path=os.path.join(cwd, experiment.base_dir, artifact_namepath),
    num_classes=NUM_CLASSES,
    download_required_code=False,
    checkpoint_num_classes=len(experiment.get_log("labelmap").data),
)

# experiment parameters
train_params = {
    # ENABLING SILENT MODE
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": initial_lr,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "phase_callbacks": [SaveTrainingMetrics()],
    "max_epochs": max_epochs,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=NUM_CLASSES,
        reg_max=16,
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=NUM_CLASSES,
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7,
            ),
        )
    ],
    "metric_to_watch": "mAP@0.50",
}

# log labelmap
labels = experiment.get_dataset("test").list_labels()
experiment.log("labelmap", labelmap, "labelmap", replace=True)

# launch the training
trainer.train(
    model=model,
    training_params=train_params,
    train_loader=train_data,
    valid_loader=val_data,
)

# get the best trained model
checkpoint_path = os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME, "ckpt_best.pth")
best_model = models.get(
    MODEL_NAME, num_classes=NUM_CLASSES, checkpoint_path=os.path.join(checkpoint_path)
)

# save model to onnx format
onnx_model_path = "yolo_nas_s_checkpoints.onnx"
models.convert_to_onnx(
    model=best_model, input_shape=(3, 640, 640), out_path=onnx_model_path
)

# store both best model in .pth and onnx format to picsellia
experiment.store("model-latest", os.path.join(cwd, onnx_model_path))
experiment.store("checkpoints_best", checkpoint_path)

test_results = trainer.test(
    model=best_model,
    test_loader=test_data,
    test_metrics_list=DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        num_cls=NUM_CLASSES,
        normalize_targets=True,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7
        ),
    ),
)
# convert test metrics to types we can log to experiment
test_results = yolonas_utils.format_test_results(test_results)
experiment.log(name="test metrics", type=LogType.TABLE, data=test_results)

# log evaluations
test_image_list = experiment.get_dataset("test").list_assets()
for asset in tqdm(test_image_list, desc="Logging evaluations into Evaluations tab"):
    asset_predictions = yolonas_utils.get_asset_predictions(
        experiment, best_model, asset, conf_threshold=0.3, dataset_type="test"
    )
    bbox_list = yolonas_utils.format_asset_predictions_for_eval(
        asset_predictions, labels
    )

    if len(bbox_list) > 0:
        experiment.add_evaluation(asset, "replace", rectangles=bbox_list)
