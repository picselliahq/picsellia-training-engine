import logging
import os

import torch

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger("picsellia").setLevel(logging.INFO)

import picsellia
from picsellia import Experiment


# 1 - Get Experiment
def get_experiment() -> Experiment:
    if "api_token" not in os.environ:
        raise Exception("You must set an api_token to run this image")
    api_token = os.environ["api_token"]

    if "host" not in os.environ:
        host = "https://app.picsellia.com"
    else:
        host = os.environ["host"]

    if "organization_id" not in os.environ:
        organization_id = None
    else:
        organization_id = os.environ["organization_id"]

    client = picsellia.Client(
        api_token=api_token, host=host, organization_id=organization_id
    )

    if "experiment_id" in os.environ:
        experiment_id = os.environ["experiment_id"]

        experiment = client.get_experiment_by_id(experiment_id)
    else:
        raise Exception("You must set the experiment_id")
    return experiment


experiment = get_experiment()

# 2 - Get the artifacts
from pycocotools.coco import COCO
import json
from picsellia.exceptions import ResourceNotFoundError
from utils import create_yolo_detection_label

experiment.download_artifacts(with_tree=True)
current_dir = os.path.join(os.getcwd(), experiment.base_dir)
base_imgdir = experiment.png_dir

parameters = experiment.get_log(name="parameters").data
attached_datasets = experiment.list_attached_dataset_versions()

if len(attached_datasets) == 3:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')"
        )
    try:
        test_ds = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')"
        )
    try:
        val_ds = experiment.get_dataset(name="val")
    except Exception:
        try:
            val_ds = experiment.get_dataset(name="eval")
        except Exception:
            raise ResourceNotFoundError(
                "Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                expecting 'train', 'test', ('val' or 'eval')"
            )

    labels = train_ds.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i): label.name for i, label in enumerate(labels)}

    for data_type, dataset in {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }.items():
        coco_annotation = dataset.build_coco_file_locally(
            enforced_ordered_categories=label_names
        )
        annotations_dict = coco_annotation.dict()
        annotations_path = os.path.join(base_imgdir, f"{data_type}_annotations.json")

        with open(annotations_path, "w") as f:
            f.write(json.dumps(annotations_dict))

        annotations_coco = COCO(annotations_path)

        dataset.list_assets().download(
            target_path=os.path.join(base_imgdir, data_type, "images"), max_workers=8
        )

else:
    dataset = experiment.list_attached_dataset_versions()[0]
    coco_annotation = dataset.build_coco_file_locally()
    annotations_dict = coco_annotation.dict()
    annotations_path = "annotations.json"
    with open(annotations_path, "w") as f:
        f.write(json.dumps(annotations_dict))
    annotations_coco = COCO(annotations_path)

    labels = dataset.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i): label.name for i, label in enumerate(labels)}

    prop = (
        0.7
        if not "prop_train_split" in parameters.keys()
        else parameters["prop_train_split"]
    )

    train_assets, test_assets, val_assets, _, _, _, _ = dataset.train_test_val_split(
        ratios=[prop, (1.0 - prop) / 2, (1.0 - prop) / 2]
    )

    for data_type, assets in {
        "train": train_assets,
        "val": val_assets,
        "test": test_assets,
    }.items():
        assets.download(
            target_path=os.path.join(base_imgdir, data_type, "images"), max_workers=8
        )

        create_yolo_detection_label(
            experiment, data_type, annotations_dict, annotations_coco, label_names
        )

# 3 - Log the labelmap
experiment.log("labelmap", labelmap, "labelmap", replace=True)
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

if "architecture" not in parameters:
    raise AssertionError(
        'The YoloX cannot be created since the model parameter "architecture" is missing.'
    )

model_architecture = parameters["architecture"]
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

# 6 - Launch the training
from YOLOX.tools.train import main, make_parser
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

args.num_classes = len(labelmap)
args.learning_rate = learning_rate
args.batch_size = batch_size
args.epochs = epochs
args.image_size = (image_size, image_size)
args.picsellia_experiment = experiment

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

# 7 - Artifact storing
model = exp.get_model()

file_name = os.path.join(exp.output_dir, args.experiment_name)
ckpt_file = os.path.join(file_name, "best_ckpt.pth")

# load the model state dict
ckpt = torch.load(ckpt_file, map_location="cpu")

model.eval()
model.load_state_dict(ckpt["model"])
model = model.to("cpu")
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
