import glob
import logging
import os

from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import tqdm
import yaml
from picsellia.types.enums import InferenceType
from pycocotools.coco import COCO
import torch
import re


class YOLOv(Enum):
    V8 = "V8"
    V7 = "V7"
    V5 = "V5"


class YOLOFormatter:
    def __init__(
        self,
        fpath: str,
        imdir: str,
        mode: InferenceType,
        steps=["train", "test", "val"],
    ) -> None:
        """
        fpath (str): path to COCO .json file
        imdir (str): path to your images folder
        targetdir (str): path the target dir for the final YOLO formatted dataset.
        mode (InferenceType): "OBJECT_DETECTION", "SEGMENTATION", "CLASSIFICATION"
        """
        self.fpath = fpath
        self.imdir = imdir
        self.mode = mode
        self.steps = [steps] if isinstance(steps, str) else steps

    def __countList(self, lst1, lst2):
        return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]

    def _coco_poly2yolo_poly(self, ann, im_w, im_h) -> List[float]:
        pair_index = np.arange(0, len(ann), 2)
        impair_index = np.arange(1, len(ann), 2)
        Xs = list(map(ann.__getitem__, pair_index))
        xs = list(map(lambda x: x / im_w, Xs))
        Ys = list(map(ann.__getitem__, impair_index))
        ys = list(map(lambda x: x / im_h, Ys))
        return self.__countList(xs, ys)

    def _coco_bbox2yolo_bbox(self, ann, im_w, im_h) -> List[float]:
        x1, y1, w, h = ann["bbox"]
        return [
            ((2 * x1 + w) / (2 * im_w)),
            ((2 * y1 + h) / (2 * im_h)),
            w / im_w,
            h / im_h,
        ]

    def _coco_classif2yolo_classif(self, ann, im_w, im_h):
        return []

    def coco2yolo(self, ann, im_w, im_h) -> callable:
        if self.mode == InferenceType.OBJECT_DETECTION:
            return self._coco_bbox2yolo_bbox(ann, im_w, im_h)
        elif self.mode == InferenceType.SEGMENTATION:
            return self._coco_poly2yolo_poly(ann, im_w, im_h)
        elif self.mode == InferenceType.CLASSIFICATION:
            return self._coco_classif2yolo_classif(ann, im_w, im_h)

    def convert(self):
        assert os.path.isdir(
            os.path.join(self.imdir, "train")
        ), "you must put your images under train/test/val folders."
        assert os.path.isdir(
            os.path.join(self.imdir, "test")
        ), "you must put your images under train/test/val folders."
        assert os.path.isdir(
            os.path.join(self.imdir, "val")
        ), "you must put your images under train/test/val folders."

        for split in ["train", "test", "val"]:
            self.coco = COCO(self.fpath)
            logging.info(f"Formatting {split} folder ..")
            dataset_path = os.path.join(self.imdir, split)
            image_filenames = os.listdir(os.path.join(dataset_path, "images"))
            labels_path = os.path.join(dataset_path, "labels")
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            for img in tqdm.tqdm(self.coco.loadImgs(self.coco.imgs)):
                result = []
                if (
                    img["file_name"] in image_filenames
                ):  # check if image is inside your folder first
                    txt_name = img["file_name"][:-4] + ".txt"
                    for ann in self.coco.loadAnns(
                        self.coco.getAnnIds(imgIds=img["id"])
                    ):
                        line = " ".join(
                            [
                                str(x)
                                for x in self.coco2yolo(
                                    ann, img["width"], img["height"]
                                )
                            ]
                        )
                        result.append(f"{ann['category_id']} {line}")
                    with open(os.path.join(labels_path, txt_name), "w") as f:
                        f.write("\n".join(result))

    def generate_yaml(self, dpath: str = "data.yaml") -> str:
        names = [label["name"] for label in self.coco.loadCats(self.coco.cats)]
        data_config = {
            "train": os.path.join(self.imdir, "train"),
            "val": os.path.join(self.imdir, "val"),
            "test": os.path.join(self.imdir, "test"),
            "nc": len(names),
            "names": names,
        }
        f = open(dpath, "w+")
        yaml.dump(data_config, f, allow_unicode=True)
        return dpath


def get_latest_file(path, run_type: InferenceType):
    if run_type == InferenceType.OBJECT_DETECTION:
        run_type = "detect"
    elif run_type == InferenceType.SEGMENTATION:
        run_type = "segment"
    elif run_type == InferenceType.CLASSIFICATION:
        run_type == "classify"
    else:
        raise ValueError("invalide run_type")
    """Returns the name of the latest (most recent) file 
    of the joined path(s)"""
    fullpath = os.path.join(path, run_type, "*")
    list_of_files = glob.glob(fullpath)  # You may use iglob in Python3
    if not list_of_files:  # I prefer using the negation
        return None  # because it behaves like a shortcut
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)
    return os.path.join(path, run_type, filename)


def get_train_infos(run_type: InferenceType):
    last_run_path = get_latest_file("runs", run_type)
    weights_path = os.path.join(last_run_path, "weights", "best.pt")
    results_path = os.path.join(last_run_path, "results.csv")
    return weights_path, results_path


def edit_model_yaml(label_map, experiment_name, config_path=None):
    for path in os.listdir(config_path):
        if path.endswith("yaml"):
            ymlpath = os.path.join(config_path, path)
    path = Path(ymlpath)
    with open(ymlpath, "r") as f:
        data = f.readlines()

    temp = re.findall(r"\d+", data[3])
    res = list(map(int, temp))

    data[3] = data[3].replace(str(res[0]), str(len(label_map)))

    if config_path is None:
        opath = (
            "."
            + ymlpath.split(".")[1]
            + "_"
            + experiment_name
            + "."
            + ymlpath.split(".")[2]
        )
    else:
        opath = (
            "./"
            + ymlpath.split(".")[0]
            + "_"
            + experiment_name
            + "."
            + ymlpath.split(".")[1]
        )
    with open(opath, "w") as f:
        for line in data:
            f.write(line)

    if config_path is None:
        tmp = opath.replace("./yolov5", ".")

    else:
        tmp = (
            ymlpath.split(".")[0] + "_" + experiment_name + "." + ymlpath.split(".")[1]
        )

    return tmp


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
    opt.patience = int(params.get("patience", 100))
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
