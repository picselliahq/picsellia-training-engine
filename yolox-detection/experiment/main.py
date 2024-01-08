import logging
import os

import picsellia
import torch
from picsellia.exceptions import ResourceNotFoundError

from YOLOX.tools.demo import Predictor
from evaluator.framework_formatter import YoloFormatter
from evaluator.type_formatter import DetectionFormatter
from utils import get_experiment, evaluate_model, extract_dataset

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

# 1 - Get Experiment
experiment = get_experiment()

# 2 - Get the artifacts
experiment.download_artifacts(with_tree=True)
current_dir = os.path.join(os.getcwd(), experiment.base_dir)
train_ds, test_ds, val_ds = extract_dataset(experiment=experiment)

# 3 - Log the labelmap
logged_labelmap = {str(i): label.name for i, label in enumerate(train_ds.list_labels())}
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
parameters = experiment.get_log("parameters").data

dataset_base_dir_path = experiment.base_dir
dataset_train_annotation_file_path = f"{experiment.png_dir}/train_annotations.json"
dataset_test_annotation_file_path = f"{experiment.png_dir}/test_annotations.json"
dataset_val_annotation_file_path = f"{experiment.png_dir}/val_annotations.json"
model_latest_checkpoint_path = f"{experiment.checkpoint_dir}/{model_architecture}.pth"

if not os.path.isfile(model_latest_checkpoint_path):
    model_latest_checkpoint_path = None

learning_rate = parameters.get("learning_rate", 0.01 / 64)
batch_size = parameters.get("batch_size", 8)
epochs = parameters.get("epochs", 100)
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
    ckpt_file = os.path.join(file_name, "latest_ckpt.pth")

ckpt = torch.load(ckpt_file, map_location="cpu")
model.load_state_dict(ckpt["model"])

model.to("cpu")
model.eval()

# 8 - Run the evaluation
test_labels = test_ds.list_labels()
test_label_names = [label.name for label in test_labels]
test_labelmap = {i: label for i, label in enumerate(test_labels)}

framework_formatter = YoloFormatter(labelmap=test_labelmap)
type_formatter = DetectionFormatter(framework_formatter=framework_formatter)
yolox_predictor = Predictor(model=model, exp=exp, cls_names=test_label_names)

compute_metrics_job = evaluate_model(
    yolox_predictor=yolox_predictor,
    type_formatter=type_formatter,
    experiment=experiment,
    dataset=test_ds,
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

# 10 - Finally, wait for the metrics computation
try:
    compute_metrics_job.wait_for_done()

except picsellia.exceptions.WaitingAttemptsTimeout:
    logging.info(
        "The compute metrics job has reached its timeout."
        "While the job will continue running in the background, the training script will now terminate."
    )
