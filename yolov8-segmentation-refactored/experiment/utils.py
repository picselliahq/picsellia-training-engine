import json
import logging
import os
from collections import OrderedDict
from typing import Optional, Tuple
import numpy as np
import torch
import yaml
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import LogType
from pycocotools.coco import COCO
from ultralytics import YOLO


def get_train_test_eval_datasets_from_experiment(
    experiment: Experiment,
) -> (
    tuple[bool, DatasetVersion, DatasetVersion, DatasetVersion]
    | tuple[bool, None, None, None]
):
    number_of_attached_datasets = len(experiment.list_attached_dataset_versions())
    has_three_datasets = False
    if number_of_attached_datasets == 3:
        has_three_datasets = True
        train_set, test_set, eval_set = _get_three_attached_datasets(experiment)
    elif number_of_attached_datasets == 1:
        logging.info(
            "We only found one dataset inside your experiment, the train/test/split will be performed automatically."
        )
        train_set: DatasetVersion = experiment.list_attached_dataset_versions()[0]
        test_set = None
        eval_set = None

    else:
        logging.info("We need either 1 or 2 datasets attached to this experiment ")
        train_set, test_set, eval_set = None, None, None

    return has_three_datasets, train_set, test_set, eval_set


def _get_three_attached_datasets(
    experiment: Experiment,
) -> tuple[DatasetVersion, DatasetVersion, DatasetVersion]:
    try:
        train_set = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', 'eval')"
        )
    try:
        test_set = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                            expecting 'train', 'test', 'eval')"
        )
    try:
        eval_set = experiment.get_dataset(name="eval")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                expecting 'train', 'test', 'eval')"
        )
    return train_set, test_set, eval_set


def write_annotation_file(annotations_dict: dict, annotations_path: str):
    with open(annotations_path, "w") as f:
        f.write(json.dumps(annotations_dict))


def create_yolo_segmentation_label(
    experiment: Experiment,
    data_type: str,
    annotations_dict: dict,
    annotations_coco: COCO,
    label_names: list,
):
    dataset_path = os.path.join(experiment.png_dir, data_type)
    image_filenames = os.listdir(os.path.join(dataset_path, "images"))

    labels_path = os.path.join(dataset_path, "labels")

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for img in annotations_dict["images"]:
        img_filename = img["file_name"]
        if img_filename in image_filenames:
            create_img_label_segmentation(
                img, annotations_coco, labels_path, label_names
            )


def create_img_label_segmentation(
    image: dict, annotations_coco: COCO, labels_path: str, label_names: list
):
    result = []
    img_id = image["id"]
    img_filename = image["file_name"]
    w = image["width"]
    h = image["height"]
    txt_name = os.path.splitext(img_filename)[0] + ".txt"
    annotation_ids = annotations_coco.getAnnIds(imgIds=img_id)
    anns = annotations_coco.loadAnns(annotation_ids)
    for ann in anns:
        seg = coco_to_yolo_segmentation(ann["segmentation"], w, h)
        seg_string = " ".join([str(x) for x in seg])
        label = label_names.index(
            annotations_coco.loadCats(ann["category_id"])[0]["name"]
        )
        result.append(f"{label} {seg_string}")
    with open(os.path.join(labels_path, txt_name), "w") as f:
        f.write("\n".join(result))


def coco_to_yolo_segmentation(ann: list, image_w: int, image_h: int) -> list:
    pair_index = np.arange(0, len(ann[0]), 2)
    impair_index = np.arange(1, len(ann[0]), 2)
    Xs = list(map(ann[0].__getitem__, pair_index))
    xs = list(map(lambda x: x / image_w, Xs))
    Ys = list(map(ann[0].__getitem__, impair_index))
    ys = list(map(lambda x: x / image_h, Ys))
    return interleave_lists(xs, ys)


def interleave_lists(lst1: list, lst2: list) -> list:
    return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]


def get_prop_parameter(parameters: dict) -> float:
    return float(parameters.get("prop_train_split", 0.7))


def log_split_dataset_repartition_to_experiment(
    experiment: Experiment,
    labelmap: dict,
    train_rep: dict[str, list],
    test_rep: dict[str, list],
    val_rep: dict[str, list],
) -> None:
    experiment.log(
        "train-split",
        order_repartition_according_labelmap(labelmap, train_rep),
        "bar",
        replace=True,
    )
    experiment.log(
        "test-split",
        order_repartition_according_labelmap(labelmap, test_rep),
        "bar",
        replace=True,
    )
    experiment.log(
        "val-split",
        order_repartition_according_labelmap(labelmap, val_rep),
        "bar",
        replace=True,
    )


def order_repartition_according_labelmap(labelmap: dict, repartition: dict) -> dict:
    ordered_rep = {"x": list(labelmap.values()), "y": []}
    for name in ordered_rep["x"]:
        ordered_rep["y"].append(repartition["y"][repartition["x"].index(name)])
    return ordered_rep


def generate_data_yaml(experiment: Experiment, labelmap: dict, config_path: str) -> str:
    cwd = os.getcwd()

    if not os.path.exists(config_path):
        os.makedirs(config_path)
    data_config_path = os.path.join(config_path, "data_config.yaml")
    n_classes = len(labelmap)
    labelmap = {int(k): v for k, v in labelmap.items()}
    ordered_labelmap = dict(sorted(OrderedDict(labelmap).items()))
    data_config = {
        "train": os.path.join(cwd, experiment.png_dir, "train"),
        "val": os.path.join(cwd, experiment.png_dir, "val"),
        "test": os.path.join(cwd, experiment.png_dir, "test"),
        "nc": n_classes,
        "names": list(ordered_labelmap.values()),
    }
    with open(data_config_path, "w+") as f:
        yaml.dump(data_config, f, allow_unicode=True)
    return data_config_path


def send_run_to_picsellia(
    experiment: Experiment, cwd: str, save_dir: str = None, imgsz: int = 640
):
    final_run_path = save_dir if save_dir is not None else find_final_run(cwd)
    best_weights, hyp_yaml = get_weights_and_config(final_run_path)
    model_latest_path = find_model_latest_path(final_run_path, imgsz)

    if model_latest_path:
        experiment.store("model-latest", model_latest_path)
    if best_weights:
        experiment.store("checkpoint-index-latest", best_weights)
    if hyp_yaml:
        experiment.store("checkpoint-data-latest", hyp_yaml)

    for curve in get_metrics_curves(final_run_path):
        if curve:
            name = extract_file_name(curve)
            experiment.log(name, curve, LogType.IMAGE)

    for batch in get_batch_mosaics(final_run_path):
        if batch:
            name = extract_file_name(batch)
            experiment.log(name, batch, LogType.IMAGE)


def find_final_run(cwd: str) -> str:
    runs_path = os.path.join(cwd, "runs", "train")
    dirs = os.listdir(runs_path)
    dirs.sort()
    if len(dirs) == 1:
        return os.path.join(runs_path, dirs[0])
    base = dirs[0][:7]
    truncate_dirs = [n[len(base) - 1 :] for n in dirs]
    last_run_nb = max(truncate_dirs)[-1]
    if last_run_nb == "p":
        last_run_nb = ""
    return os.path.join(runs_path, base + last_run_nb)


def get_weights_and_config(final_run_path: str) -> tuple[str | None, str | None]:
    best_weights = None
    hyp_yaml = None
    best_weights_path = os.path.join(final_run_path, "weights", "best.pt")
    hyp_yaml_path = os.path.join(final_run_path, "hyp.yaml")
    args_yaml_path = os.path.join(final_run_path, "args.yaml")

    if os.path.isfile(best_weights_path):
        best_weights = best_weights_path
    elif os.path.isfile(hyp_yaml_path):
        hyp_yaml = hyp_yaml_path
    elif os.path.isfile(args_yaml_path):
        hyp_yaml = args_yaml_path

    return best_weights, hyp_yaml


def find_model_latest_path(final_run_path: str, imgsz: int) -> str | None:
    model_dir = os.path.join(final_run_path, "weights")
    model_paths = [
        os.path.join(model_dir, "best.onnx"),
        os.path.join(model_dir, "last.onnx"),
        os.path.join(model_dir, "best.pt"),
        os.path.join(model_dir, "last.pt"),
    ]

    for model_path in model_paths:
        if os.path.isfile(model_path):
            if model_path.endswith(".pt"):
                checkpoint_path = model_path
                model = YOLO(checkpoint_path)
                model.export(format="onnx", imgsz=imgsz, task="segment")
                return os.path.join(final_run_path, "weights", "best.onnx")
            return model_path

    logging.warning("Can't find last checkpoints to be uploaded")
    return None


def get_metrics_curves(
    final_run_path: str,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]:
    confusion_matrix_path = os.path.join(final_run_path, "confusion_matrix.png")
    F1_curve_path = os.path.join(final_run_path, "F1_curve.png")
    labels_correlogram_path = os.path.join(final_run_path, "labels_correlogram.jpg")
    labels_path = os.path.join(final_run_path, "labels.jpg")
    P_curve_path = os.path.join(final_run_path, "P_curve.png")
    PR_curve_path = os.path.join(final_run_path, "PR_curve.png")
    R_curve_path = os.path.join(final_run_path, "R_curve.png")
    BoxF1_curve_path = os.path.join(final_run_path, "BoxF1_curve.png")
    BoxP_curve_path = os.path.join(final_run_path, "BoxP_curve.png")
    BoxPR_curve_path = os.path.join(final_run_path, "BoxPR_curve.png")
    BoxR_curve_path = os.path.join(final_run_path, "BoxR_curve.png")
    MaskF1_curve_path = os.path.join(final_run_path, "MaskF1_curve.png")
    MaskP_curve_path = os.path.join(final_run_path, "MaskP_curve.png")
    MaskPR_curve_path = os.path.join(final_run_path, "MaskPR_curve.png")
    MaskR_curve_path = os.path.join(final_run_path, "MaskR_curve.png")

    confusion_matrix = (
        confusion_matrix_path if os.path.isfile(confusion_matrix_path) else None
    )
    F1_curve = F1_curve_path if os.path.isfile(F1_curve_path) else None
    labels_correlogram = (
        labels_correlogram_path if os.path.isfile(labels_correlogram_path) else None
    )
    labels = labels_path if os.path.isfile(labels_path) else None
    P_curve = P_curve_path if os.path.isfile(P_curve_path) else None
    PR_curve = PR_curve_path if os.path.isfile(PR_curve_path) else None
    R_curve = R_curve_path if os.path.isfile(R_curve_path) else None
    BoxF1_curve = BoxF1_curve_path if os.path.isfile(BoxF1_curve_path) else None
    BoxP_curve = BoxP_curve_path if os.path.isfile(BoxP_curve_path) else None
    BoxPR_curve = BoxPR_curve_path if os.path.isfile(BoxPR_curve_path) else None
    BoxR_curve = BoxR_curve_path if os.path.isfile(BoxR_curve_path) else None
    MaskF1_curve = MaskF1_curve_path if os.path.isfile(MaskF1_curve_path) else None
    MaskP_curve = MaskP_curve_path if os.path.isfile(MaskP_curve_path) else None
    MaskPR_curve = MaskPR_curve_path if os.path.isfile(MaskPR_curve_path) else None
    MaskR_curve = MaskR_curve_path if os.path.isfile(MaskR_curve_path) else None

    return (
        confusion_matrix,
        F1_curve,
        labels_correlogram,
        labels,
        P_curve,
        PR_curve,
        R_curve,
        BoxF1_curve,
        BoxP_curve,
        BoxPR_curve,
        BoxR_curve,
        MaskF1_curve,
        MaskP_curve,
        MaskPR_curve,
        MaskR_curve,
    )


def extract_file_name(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def get_batch_mosaics(
    final_run_path: str,
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    val_batch0_labels = None
    val_batch0_pred = None
    val_batch1_labels = None
    val_batch1_pred = None
    val_batch2_labels = None
    val_batch2_pred = None

    val_batch0_labels_path = os.path.join(final_run_path, "val_batch0_labels.jpg")
    val_batch0_pred_path = os.path.join(final_run_path, "val_batch0_pred.jpg")
    val_batch1_labels_path = os.path.join(final_run_path, "val_batch1_labels.jpg")
    val_batch1_pred_path = os.path.join(final_run_path, "val_batch1_pred.jpg")
    val_batch2_labels_path = os.path.join(final_run_path, "val_batch2_labels.jpg")
    val_batch2_pred_path = os.path.join(final_run_path, "val_batch2_pred.jpg")

    if os.path.isfile(val_batch0_labels_path):
        val_batch0_labels = val_batch0_labels_path
    if os.path.isfile(val_batch0_pred_path):
        val_batch0_pred = val_batch0_pred_path
    if os.path.isfile(val_batch1_labels_path):
        val_batch1_labels = val_batch1_labels_path
    if os.path.isfile(val_batch1_pred_path):
        val_batch1_pred = val_batch1_pred_path
    if os.path.isfile(val_batch2_labels_path):
        val_batch2_labels = val_batch2_labels_path
    if os.path.isfile(val_batch2_pred_path):
        val_batch2_pred = val_batch2_pred_path

    return (
        val_batch0_labels,
        val_batch0_pred,
        val_batch1_labels,
        val_batch1_pred,
        val_batch2_labels,
        val_batch2_pred,
    )


def setup_hyp(
    experiment: Experiment = None,
    data_yaml_path: str = None,
    params: dict = None,
    cwd: str = None,
    task: str = "detect",
):
    if params is None:
        params = {}
    tmp = os.listdir(experiment.checkpoint_dir)

    for f in tmp:
        if f.endswith(".pt"):
            weight_path = os.path.join(experiment.checkpoint_dir, f)
        if f.endswith(".yaml"):
            hyp_path = os.path.join(experiment.checkpoint_dir, f)

    opt = Opt()

    opt.task = task
    opt.mode = "train"
    opt.cwd = cwd
    # Train settings -------------------------------------------------------------------------------------------------------
    opt.model = weight_path
    opt.data = data_yaml_path
    opt.epochs = int(params.get("epochs", 100))
    opt.patience = int(params.get("patience", 500))
    opt.batch = int(params.get("batch_size", 4))
    opt.imgsz = int(params.get("input_shape", 640))
    opt.save = bool(params.get("save", True))
    opt.save_period = int(params.get("save_period", 100))
    opt.cache = bool(params.get("cache", False))
    opt.device = "0" if torch.cuda.is_available() else "cpu"
    opt.workers = int(params.get("workers", 8))
    opt.project = cwd
    opt.name = "exp"
    opt.exist_ok = bool(params.get("exist_ok", False))
    opt.pretrained = params.get("pretrained", True)
    opt.optimizer = params.get("optimizer", "Adam")
    opt.verbose = params.get("verbose", True)
    opt.seed = int(params.get("seed", 0))
    opt.deterministic = bool(params.get("deterministic", True))
    opt.single_cls = bool(params.get("single_cls", False))
    opt.image_weights = bool(params.get("image_weights", False))
    opt.rect = bool(params.get("rect", False))
    opt.cos_lr = bool(params.get("cos_lr", False))  # use cosine learning rate scheduler
    opt.close_mosaic = int(
        params.get("close_mosaic", 10)
    )  # Disable mosaic augmentation for the final N epochs
    opt.resume = bool(
        params.get("resume", False)
    )  # Resume training from the last checkpoint
    opt.min_memory = bool(
        params.get("min_memory", False)
    )  # Minimize memory footprint for the loss function

    # Segmentation
    opt.overlap_mask = bool(
        params.get("overlap_mask", True)
    )  # Masks should overlap during training (segment train only)
    opt.mask_ratio = int(
        params.get("mask_ratio", 4)
    )  # Mask downsample ratio (segment train only)

    # Classification
    opt.dropout = float(params.get("dropout", 0.0))

    # Val/Test settings ----------------------------------------------------------------------------------------------------
    opt.val = bool(params.get("val", True))  # Validate/test during training
    opt.split = str(
        params.get("split", "val")
    )  # Dataset split to use for validation, e.g., 'val', 'test', or 'train'
    opt.save_json = bool(params.get("save_json", False))  # Save results to JSON file
    opt.save_hybrid = bool(
        params.get("save_hybrid", False)
    )  # Save hybrid version of labels (labels + additional predictions)
    opt.conf = float(
        params.get("conf", 0.25)
    )  # Object confidence threshold for detection (default 0.25 for predict, 0.001 for val)
    opt.iou = float(
        params.get("iou", 0.7)
    )  # Intersection over Union (IoU) threshold for NMS
    opt.max_det = int(
        params.get("max_det", 300)
    )  # Maximum number of detections per image
    opt.half = bool(params.get("half", False))  # Use half precision (FP16)
    opt.dnn = bool(params.get("dnn", False))  # Use OpenCV DNN for ONNX inference
    opt.plots = bool(params.get("plots", True))  # Save plots during train/val

    # Prediction settings --------------------------------------------------------------------------------------------------
    opt.source = str(params.get("source", ""))  # Source directory for images or videos
    opt.show = bool(params.get("show", False))  # Show results if possible
    opt.save_txt = bool(params.get("save_txt", False))  # Save results as .txt file
    opt.save_conf = bool(
        params.get("save_conf", False)
    )  # Save results with confidence scores
    opt.save_crop = bool(
        params.get("save_crop", False)
    )  # Save cropped images with results
    opt.hide_labels = bool(params.get("hide_labels", False))  # Hide labels
    opt.hide_conf = bool(params.get("hide_conf", False))  # Hide confidence scores
    opt.vid_stride = int(params.get("vid_stride", 1))  # Video frame-rate stride
    opt.line_thickness = int(
        params.get("line_thickness", 3)
    )  # Bounding box thickness (pixels)
    opt.visualize = bool(params.get("visualize", False))  # Visualize model features
    opt.augment = bool(
        params.get("augment", False)
    )  # Apply image augmentation to prediction sources
    opt.agnostic_nms = bool(params.get("agnostic_nms", False))  # Class-agnostic NMS
    # opt.classes= # Filter results by class, e.g., class=0, or class=[0,2,3]
    opt.retina_masks = bool(
        params.get("retina_masks", False)
    )  # Use high-resolution segmentation masks
    opt.boxes = bool(
        params.get("boxes", True)
    )  # Show boxes in segmentation predictions

    # Export settings ------------------------------------------------------------------------------------------------------
    opt.format = str(params.get("format", "torchscript"))  # Format to export to
    opt.keras = bool(params.get("keras", False))  # Use Keras
    opt.optimize = bool(
        params.get("optimize", False)
    )  # TorchScript=optimize for mobile
    opt.int8 = bool(params.get("int8", False))  # CoreML/TF INT8 quantization
    opt.dynamic = bool(params.get("dynamic", False))  # ONNX/TF/TensorRT=dynamic axes
    opt.simplify = bool(params.get("simplify", False))  # ONNX: simplify model
    opt.workspace = int(params.get("workspace", 4))  # TensorRT: workspace size (GB)
    opt.nms = bool(params.get("nms", False))  # CoreML: add NMS
    # Hyperparameters ------------------------------------------------------------------------------------------------------
    opt.lr0 = float(
        params.get("lr0", 0.01)
    )  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    opt.lrf = float(params.get("lrf", 0.01))  # final learning rate (lr0 * lrf)
    opt.momentum = float(params.get("momentum", 0.937))  # SGD momentum/Adam beta1
    opt.weight_decay = float(
        params.get("weight_decay", 0.0005)
    )  # optimizer weight decay 5e-4
    opt.warmup_epochs = float(
        params.get("warmup_epochs", 3.0)
    )  # warmup epochs (fractions ok)
    opt.warmup_momentum = float(
        params.get("warmup_momentum", 0.8)
    )  # warmup initial momentum
    opt.warmup_bias_lr = float(
        params.get("warmup_bias_lr", 0.1)
    )  # warmup initial bias lr
    opt.box = float(params.get("box", 7.5))  # box loss gain
    opt.cls = float(params.get("cls", 0.5))  # cls loss gain (scale with pixels)
    opt.dfl = float(params.get("dfl", 1.5))  # dfl loss gain
    opt.fl_gamma = float(
        params.get("fl_gamma", 0.0)
    )  # focal loss gamma (efficientDet default gamma=1.5)
    opt.label_smoothing = float(
        params.get("label_smoothing", 0.0)
    )  # label smoothing (fraction)
    opt.nbs = int(params.get("nbs", 64))  # nominal batch size
    opt.hsv_h = float(
        params.get("hsv_h", 0.015)
    )  # image HSV-Hue augmentation (fraction)
    opt.hsv_s = float(
        params.get("hsv_s", 0.7)
    )  # image HSV-Saturation augmentation (fraction)
    opt.hsv_v = float(
        params.get("hsv_v", 0.4)
    )  # image HSV-Value augmentation (fraction)
    opt.degrees = float(params.get("degrees", 0.0))  # image rotation (+/- deg)
    opt.translate = float(
        params.get("translate", 0.1)
    )  # image translation (+/- fraction)
    opt.scale = float(params.get("scale", 0.5))  # image scale (+/- gain)
    opt.shear = float(params.get("shear", 0.0))  # image shear (+/- deg)
    opt.perspective = float(
        params.get("perspective", 0.0)
    )  # image perspective (+/- fraction), range 0-0.001
    opt.flipud = float(params.get("flipud", 0.0))  # image flip up-down (probability)
    opt.fliplr = float(params.get("fliplr", 0.5))  # image flip left-right (probability)
    opt.mosaic = float(params.get("mosaic", 1.0))  # image mosaic (probability)
    opt.mixup = float(params.get("mixup", 0.0))  # image mixup (probability)
    opt.copy_paste = float(
        params.get("copy_paste", 0.0)
    )  # segment copy-paste (probability)
    return opt


class Opt:
    pass
