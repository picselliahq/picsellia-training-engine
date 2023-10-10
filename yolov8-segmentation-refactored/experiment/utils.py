import json
import logging
import os
import yaml
import torch
from collections import OrderedDict
from ultralytics import YOLO

import numpy as np
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import LogType


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
    exp, data_type, annotations_dict, annotations_coco, label_names
):
    dataset_path = os.path.join(exp.png_dir, data_type)
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


def create_img_label_segmentation(img, annotations_coco, labels_path, label_names):
    result = []
    img_id = img["id"]
    img_filename = img["file_name"]
    w = img["width"]
    h = img["height"]
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


def coco_to_yolo_segmentation(ann, image_w, image_h):
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


def generate_data_yaml(experiment: Experiment, labelmap: dict, config_path: str):
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


def send_run_to_picsellia(experiment, cwd, save_dir=None, imgsz=640):
    if save_dir is not None:
        final_run_path = save_dir
    else:
        final_run_path = find_final_run(cwd)
    best_weigths, hyp_yaml = get_weights_and_config(final_run_path)

    model_latest_path = os.path.join(final_run_path, "weights", "best.onnx")
    model_dir = os.path.join(final_run_path, "weights")
    if os.path.isfile(os.path.join(model_dir, "best.onnx")):
        model_latest_path = os.path.join(model_dir, "best.onnx")
    elif os.path.isfile(os.path.join(model_dir, "last.onnx")):
        model_latest_path = os.path.join(model_dir, "last.onnx")
    elif os.path.isfile(os.path.join(model_dir, "best.pt")):
        checkpoint_path = os.path.join(model_dir, "best.pt")
        model = YOLO(checkpoint_path)
        model.export(format="onnx", imgsz=imgsz, task="segment")
        model_latest_path = os.path.join(final_run_path, "weights", "best.onnx")
    elif not os.path.isfile(os.path.join(model_dir, "last.pt")):
        checkpoint_path = os.path.join(model_dir, "last.pt")
        model = YOLO(checkpoint_path)
        model.export(format="onnx", imgsz=imgsz, task="segment")
        model_latest_path = os.path.join(final_run_path, "weights", "last.onnx")
    else:
        logging.warning("Can't find last checkpoints to be uploaded")
        model_latest_path = None
    if model_latest_path is not None:
        experiment.store("model-latest", model_latest_path)
    if best_weigths is not None:
        experiment.store("checkpoint-index-latest", best_weigths)
    if hyp_yaml is not None:
        experiment.store("checkpoint-data-latest", hyp_yaml)
    for curve in get_metrics_curves(final_run_path):
        if curve is not None:
            name = curve.split("/")[-1].split(".")[0]
            experiment.log(name, curve, LogType.IMAGE)
    for batch in get_batch_mosaics(final_run_path):
        if batch is not None:
            name = batch.split("/")[-1].split(".")[0]
            experiment.log(name, batch, LogType.IMAGE)


def find_final_run(cwd):
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


def get_weights_and_config(final_run_path):
    best_weights = None
    hyp_yaml = None
    if os.path.isfile(os.path.join(final_run_path, "weights", "best.pt")):
        best_weights = os.path.join(final_run_path, "weights", "best.pt")
    if os.path.isfile(os.path.join(final_run_path, "hyp.yaml")):
        hyp_yaml = os.path.join(final_run_path, "hyp.yaml")
    if os.path.isfile(os.path.join(final_run_path, "args.yaml")):
        hyp_yaml = os.path.join(final_run_path, "args.yaml")
    return best_weights, hyp_yaml


def get_metrics_curves(final_run_path):
    confusion_matrix = None
    F1_curve = None
    labels_correlogram = None
    labels = None
    P_curve = None
    PR_curve = None
    R_curve = None
    BoxF1_curve = None
    BoxP_curve = None
    BoxPR_curve = None
    BoxR_curve = None
    MaskF1_curve = None
    MaskP_curve = None
    MaskPR_curve = None
    MaskR_curve = None
    if os.path.isfile(os.path.join(final_run_path, "confusion_matrix.png")):
        confusion_matrix = os.path.join(final_run_path, "confusion_matrix.png")
    if os.path.isfile(os.path.join(final_run_path, "F1_curve.png")):
        F1_curve = os.path.join(final_run_path, "F1_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "labels_correlogram.jpg")):
        labels_correlogram = os.path.join(final_run_path, "labels_correlogram.jpg")
    if os.path.isfile(os.path.join(final_run_path, "labels.jpg")):
        labels = os.path.join(final_run_path, "labels.jpg")
    if os.path.isfile(os.path.join(final_run_path, "P_curve.png")):
        P_curve = os.path.join(final_run_path, "P_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "PR_curve.png")):
        PR_curve = os.path.join(final_run_path, "PR_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "R_curve.png")):
        R_curve = os.path.join(final_run_path, "R_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "BoxF1_curve.png")):
        BoxF1_curve = os.path.join(final_run_path, "BoxF1_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "BoxP_curve.png")):
        BoxP_curve = os.path.join(final_run_path, "BoxP_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "BoxPR_curve.png")):
        BoxPR_curve = os.path.join(final_run_path, "BoxPR_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "BoxR_curve.png")):
        BoxR_curve = os.path.join(final_run_path, "BoxR_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "MaskF1_curve.png")):
        MaskF1_curve = os.path.join(final_run_path, "MaskF1_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "MaskP_curve.png")):
        MaskP_curve = os.path.join(final_run_path, "MaskP_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "MaskPR_curve.png")):
        MaskPR_curve = os.path.join(final_run_path, "MaskPR_curve.png")
    if os.path.isfile(os.path.join(final_run_path, "MaskR_curve.png")):
        MaskR_curve = os.path.join(final_run_path, "MaskR_curve.png")
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


def get_batch_mosaics(final_run_path):
    val_batch0_labels = None
    val_batch0_pred = None
    val_batch1_labels = None
    val_batch1_pred = None
    val_batch2_labels = None
    val_batch2_pred = None
    if os.path.isfile(os.path.join(final_run_path, "val_batch0_labels.jpg")):
        val_batch0_labels = os.path.join(final_run_path, "val_batch0_labels.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch0_pred.jpg")):
        val_batch0_pred = os.path.join(final_run_path, "val_batch0_pred.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch1_labels.jpg")):
        val_batch1_labels = os.path.join(final_run_path, "val_batch1_labels.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch1_pred.jpg")):
        val_batch1_pred = os.path.join(final_run_path, "val_batch1_pred.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch2_labels.jpg")):
        val_batch2_labels = os.path.join(final_run_path, "val_batch2_labels.jpg")
    if os.path.isfile(os.path.join(final_run_path, "val_batch2_pred.jpg")):
        val_batch2_pred = os.path.join(final_run_path, "val_batch2_pred.jpg")
    return (
        val_batch0_labels,
        val_batch0_pred,
        val_batch1_labels,
        val_batch1_pred,
        val_batch2_labels,
        val_batch2_pred,
    )


def setup_hyp(
    experiment=None,
    data_yaml_path=None,
    config_path=None,
    params={},
    label_map=[],
    cwd=None,
    task="detect",
):
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
