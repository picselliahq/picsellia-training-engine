import logging
import os

import picsellia
import torch
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import LogType

from YOLOX.tools.demo import Predictor
from evaluator.framework_formatter import YoloFormatter
from evaluator.type_formatter import DetectionFormatter
from utils import (
    get_experiment,
    extract_dataset_assets,
    evaluate_model,
)

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"

model_architecture = os.environ.get("architecture", None)

if model_architecture is None:
    raise ValueError(
        "The environment variable `architecture` is mandatory."
        "You can choose between `yolox-s`, `yolox-m`, `yolox-l` or `yolox-x`"
    )
elif model_architecture not in ("yolox-s", "yolox-m", "yolox-l", "yolox-x"):
    raise ValueError(
        f"The provided model architecture {model_architecture} is not supported."
        "You can choose between `yolox-s`, `yolox-m`, `yolox-l` or `yolox-x`"
    )

logging.getLogger("picsellia").setLevel(logging.INFO)

# 1 - Get Experiment and its parameters
experiment = get_experiment()
parameters = experiment.get_log("parameters").data

# 2 - Get the artifacts
prop_train_split = parameters.get("prop_train_split", 0.7)
experiment.download_artifacts(with_tree=True)
current_dir = os.path.join(os.getcwd(), experiment.base_dir)
(
    train_assets,
    test_assets,
    val_assets,
    train_labels,
    test_labels,
    ds_type,
) = extract_dataset_assets(experiment=experiment, prop_train_split=prop_train_split)

# 3 - Log the labelmap
logged_labelmap = {str(i): label.name for i, label in enumerate(train_labels)}
experiment.log("labelmap", logged_labelmap, "labelmap", replace=True)
cwd = os.getcwd()

# 4 - Prepare the dataset structures
if os.path.isdir(f"{experiment.png_dir}/train"):
    os.rename(f"{experiment.png_dir}/train", f"{experiment.png_dir}/train2017")

if os.path.isdir(f"{experiment.png_dir}/test"):
    os.rename(f"{experiment.png_dir}/test", f"{experiment.png_dir}/test2017")

if os.path.isdir(f"{experiment.png_dir}/val"):
    os.rename(f"{experiment.png_dir}/val", f"{experiment.png_dir}/val2017")

# 5 - Get and prepare the parameters
dataset_base_dir_path = experiment.base_dir
dataset_train_annotation_file_path = f"{experiment.png_dir}/train_annotations.json"
dataset_test_annotation_file_path = f"{experiment.png_dir}/test_annotations.json"
dataset_val_annotation_file_path = f"{experiment.png_dir}/val_annotations.json"
model_latest_checkpoint_path = None

if best_epoch_number := parameters.get("best_epoch_number", None):
    model_latest_checkpoint_path = (
        f"{experiment.base_dir}/best_ckpt_{best_epoch_number}.pth"
    )
elif os.path.isfile(f"{experiment.base_dir}/last_epoch_ckpt.pth"):
    model_latest_checkpoint_path = f"{experiment.base_dir}/last_epoch_ckpt.pth"

elif os.path.isfile(f"{experiment.base_dir}/{model_architecture}.pth"):
    model_latest_checkpoint_path = f"{experiment.base_dir}/{model_architecture}.pth"

learning_rate = parameters.get("learning_rate", 0.01 / 64)
batch_size = parameters.get("batch_size", 8)
epochs = int(parameters.get("epochs", 100))
image_size = int(parameters.get("image_size", 640))
eval_interval = int(parameters.get("eval_interval", 5))

# 6 - Launch the training
from YOLOX.tools.train import make_parser, main
from YOLOX.yolox.exp import check_exp_value
from YOLOX.yolox.utils import configure_module, get_num_devices
from YOLOX.yolox.exp.build import get_exp_by_name
from YOLOX.yolox.models.network_blocks import SiLU
from YOLOX.yolox.utils import replace_module

# 6A - Args
configure_module()
args = make_parser().parse_args()

args.name = model_architecture

args.data_dir = experiment.png_dir
args.train_ann = dataset_train_annotation_file_path
args.test_ann = dataset_test_annotation_file_path
args.val_ann = dataset_val_annotation_file_path
args.ckpt = model_latest_checkpoint_path

args.num_classes = len(logged_labelmap)
args.learning_rate = learning_rate
args.batch_size = batch_size
args.epochs = epochs
args.image_size = (image_size, image_size)
args.picsellia_experiment = experiment
args.eval_interval = eval_interval

# 6B - Get model architecture
exp = get_exp_by_name(args)
exp.merge(args.opts)
check_exp_value(exp)

if not args.experiment_name:
    args.experiment_name = exp.exp_name

num_gpu = get_num_devices() if args.devices is None else args.devices
assert num_gpu <= get_num_devices()

# 6C - Launch training
main(exp, args)

# 7 - Prepare the model for inference and export
model = exp.get_model()

file_name = os.path.join(exp.output_dir, args.experiment_name)
ckpt_file = os.path.join(file_name, "best_ckpt.pth")

if not os.path.isfile(ckpt_file):
    ckpt_file = os.path.join(file_name, "last_epoch_ckpt.pth")

ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])

model.to("cpu")
model.eval()

# 8 - Run the evaluation
test_label_names = [label.name for label in test_labels]
test_labelmap = {i: label for i, label in enumerate(test_labels)}

framework_formatter = YoloFormatter(labelmap=test_labelmap)
type_formatter = DetectionFormatter(framework_formatter=framework_formatter)
yolox_predictor = Predictor(model=model, exp=exp, cls_names=test_label_names)

compute_metrics_job = evaluate_model(
    yolox_predictor=yolox_predictor,
    type_formatter=type_formatter,
    experiment=experiment,
    asset_list=test_assets,
    dataset_type=ds_type,
)

# 9 - Export the model to ONNX
model = replace_module(model, torch.nn.SiLU, SiLU)
model.head.decode_in_inference = False
dummy_input = torch.randn(1, 3, image_size, image_size)
model_path = os.path.join(exp.output_dir, args.experiment_name, "best.onnx")

torch.onnx.export(
    model,
    dummy_input,
    model_path,
)
experiment.store("model-latest", model_path)

# 9A - Handle Best checkpoint and add the epoch number
best_checkpoint_path = os.path.join(
    exp.output_dir, args.experiment_name, " best_ckpt.pth"
)
if os.path.isfile(best_checkpoint_path):
    new_best_epoch_number = exp.get_trainer().best_ap
    best_checkpoint_path_with_epoch = os.path.join(
        exp.output_dir, args.experiment_name, f"best_ckpt_{new_best_epoch_number}.pth"
    )
    experiment.log("best_epoch_number", new_best_epoch_number, LogType.VALUE)
    experiment.store("best-ckpt", best_checkpoint_path)

# 9B - Handle latest checkpoint
latest_checkpoint_path = os.path.join(
    exp.output_dir, args.experiment_name, "last_epoch_ckpt.pth"
)
if os.path.isfile(latest_checkpoint_path):
    experiment.store("last-epoch-ckpt", latest_checkpoint_path)

# 10 - Finally, wait for the metrics computation
try:
    t = 1
    compute_metrics_job.wait_for_done()

except picsellia.exceptions.WaitingAttemptsTimeout:
    logging.info(
        "The compute metrics job has reached its timeout."
        "While the job will continue running in the background, the training script will now terminate."
    )
