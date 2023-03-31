import glob
import logging
import os
import sys
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
    def __init__(self, fpath: str, imdir: str, mode: InferenceType, steps = ["train", "test", "val"]) -> None:
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
        xs = list(map(lambda x: x/im_w, Xs))
        Ys = list(map(ann.__getitem__, impair_index))
        ys = list(map(lambda x: x/im_h, Ys))
        return self.__countList(xs, ys)

    def _coco_bbox2yolo_bbox(self, ann, im_w, im_h) -> List[float]:
        x1, y1, w, h = ann["bbox"]
        return [((2*x1 + w)/(2*im_w)) , ((2*y1 + h)/(2*im_h)), w/im_w, h/im_h]

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
        assert os.path.isdir(os.path.join(self.imdir, "train")), "you must put your images under train/test/val folders."
        assert os.path.isdir(os.path.join(self.imdir, "test")), "you must put your images under train/test/val folders."
        assert os.path.isdir(os.path.join(self.imdir, "val")), "you must put your images under train/test/val folders."
        
        for split in ['train', 'test', 'val']:
            self.coco = COCO(self.fpath)
            logging.info(f"Formatting {split} folder ..")
            dataset_path = os.path.join(self.imdir, split)
            image_filenames = os.listdir(os.path.join(dataset_path, 'images'))
            labels_path = os.path.join(dataset_path, 'labels')
            if not os.path.exists(labels_path):
                os.makedirs(labels_path)
            for img in tqdm.tqdm(self.coco.loadImgs(self.coco.imgs)):
                result = []
                if img["file_name"] in image_filenames : # check if image is inside your folder first
                    txt_name = img['file_name'][:-4] + '.txt'
                    for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img['id'])):
                        line = " ".join([str(x) for x in  self.coco2yolo(ann, img['width'], img['height'])])
                        result.append(f"{ann['category_id']} {line}")
                    with open(os.path.join(labels_path, txt_name), 'w') as f:
                        f.write("\n".join(result))

    def generate_yaml(self, dpath: str = "data.yaml") -> str:
        names = [label["name"] for label in self.coco.loadCats(self.coco.cats)]
        data_config = {
            'train' : os.path.join(self.imdir, 'train'),
            'val' : os.path.join(self.imdir, 'val'),
            'test' : os.path.join(self.imdir, 'test'),
            'nc' : len(names),
            'names' : names
        }
        f = open(dpath, 'w+')
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
    fullpath = os.path.join(path,run_type, "*")
    list_of_files = glob.glob(fullpath)  # You may use iglob in Python3
    if not list_of_files:                # I prefer using the negation
        return None                      # because it behaves like a shortcut
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)
    return os.path.join(path, run_type, filename)

def get_train_infos(run_type: InferenceType):
    last_run_path = get_latest_file('runs', run_type)
    weights_path = os.path.join(last_run_path, 'weights', 'best.pt')
    results_path = os.path.join(last_run_path, 'results.csv')
    return weights_path, results_path

def edit_model_yaml(label_map, experiment_name, config_path=None):
    for path in os.listdir(config_path):
        if path.endswith('yaml'):
            ymlpath = os.path.join(config_path, path)
    path = Path(ymlpath)
    with open(ymlpath, 'r') as f:
        data = f.readlines()

    temp = re.findall(r'\d+', data[3]) 
    res = list(map(int, temp)) 

    data[3] = data[3].replace(str(res[0]), str(len(label_map)))

    if config_path is None:
        opath = '.'+ymlpath.split('.')[1]+ '_' + experiment_name+'.'+ymlpath.split('.')[2]
    else:
        opath = './'+ymlpath.split('.')[0]+'_' +experiment_name+'.'+ymlpath.split('.')[1]
    with open(opath, "w") as f:
        for line in data:
            f.write(line)

    if config_path is None:
        tmp = opath.replace('./yolov5','.')
    
    else:
        tmp = ymlpath.split('.')[0]+ '_' + experiment_name+'.'+ymlpath.split('.')[1]

    return tmp

def setup_hyp(experiment=None, data_yaml_path=None, config_path= None, params={}, label_map=[], cwd=None, task='detect'):

    tmp = os.listdir(experiment.checkpoint_dir)
    
    for f in tmp:
        if f.endswith('.pt'):
            weight_path = os.path.join(experiment.checkpoint_dir, f)
        if f.endswith('.yaml'):
            hyp_path = os.path.join(experiment.checkpoint_dir, f)
    
    opt = Opt()
    
    opt.task = task
    opt.mode = 'train'
    opt.cwd = cwd
    # Train settings -------------------------------------------------------------------------------------------------------
    opt.model = weight_path 
    opt.data = data_yaml_path
    opt.epochs = 100 if not 'epochs' in params.keys() else params["epochs"]
    opt.patience = 100 if not 'epochs' in params.keys() else params["epochs"]
    opt.batch = 4 if not 'batch_size' in params.keys() else params["batch_size"]
    opt.imgsz = 640 if not 'input_shape' in params.keys() else params["input_shape"]
    opt.save = True
    opt.save_period = 100 if not 'save_period' in params.keys() else params["save_period"]
    opt.cache = False
    opt.device = '0' if torch.cuda.is_available() else 'cpu'
    opt.workers = 8
    opt.project = cwd
    opt.name = 'exp'
    opt.exist_ok = False
    opt.pretrained = True
    opt.optimizer = 'Adam'
    opt.verbose = True
    opt.seed = 0
    opt.deterministic = True
    opt.single_cls = False
    opt.image_weights = False
    opt.rect = False
    opt.cos_lr = False  # use cosine learning rate scheduler
    opt.close_mosaic = 10  # disable mosaic augmentation for final 10 epochs
    opt.resume = False  # resume training from last checkpoint
    opt.min_memory = False  # minimize memory footprint loss function, choices=[False, True, <roll_out_thr>]
    
    # Segmentation
    opt.overlap_mask = True  # masks should overlap during training (segment train only)
    opt.mask_ratio = 4  # mask downsample ratio (segment train only)
    # Classification
    opt.dropout = 0.0
    
    # Val/Test settings ----------------------------------------------------------------------------------------------------
    opt.val = True  # validate/test during training
    opt.split = 'val'  # dataset split to use for validation, i.e. 'val', 'test' or 'train'
    opt.save_json = False  # save results to JSON file
    opt.save_hybrid = False  # save hybrid version of labels (labels + additional predictions)
    opt.conf = 0.25  # object confidence threshold for detection (default 0.25 predict, 0.001 val)
    opt.iou = 0.7  # intersection over union (IoU) threshold for NMS
    opt.max_det = 300  # maximum number of detections per image
    opt.half = False  # use half precision (FP16)
    opt.dnn = False  # use OpenCV DNN for ONNX inference
    opt.plots = True  # save plots during train/val
    
    # Prediction settings --------------------------------------------------------------------------------------------------
    opt.source='' # source directory for images or videos
    opt.show=False  # show results if possible
    opt.save_txt=False  # save results as .txt file
    opt.save_conf=False  # save results with confidence scores
    opt.save_crop=False  # save cropped images with results
    opt.hide_labels=False  # hide labels
    opt.hide_conf=False  # hide confidence scores
    opt.vid_stride=1  # video frame-rate stride
    opt.line_thickness=3  # bounding box thickness (pixels)
    opt.visualize=False  # visualize model features
    opt.augment=False  # apply image augmentation to prediction sources
    opt.agnostic_nms=False  # class-agnostic NMS
    # opt.classes= # filter results by class, i.e. class=0, or class=[0,2,3]
    opt.retina_masks=False  # use high-resolution segmentation masks
    opt.boxes=True # Show boxes in segmentation predictions
    
    # Export settings ------------------------------------------------------------------------------------------------------
    opt.format='torchscript'  # format to export to
    opt.keras=False  # use Keras
    opt.optimize=False  # TorchScript=optimize for mobile
    opt.int8=False  # CoreML/TF INT8 quantization
    opt.dynamic=False  # ONNX/TF/TensorRT=dynamic axes
    opt.simplify=False  # ONNX: simplify model
    opt.workspace=4  # TensorRT: workspace size (GB)
    opt.nms=False  # CoreML: add NMS
    
    # Hyperparameters ------------------------------------------------------------------------------------------------------
    opt.lr0=0.01  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    opt.lrf=0.01  # final learning rate (lr0 * lrf)
    opt.momentum=0.937  # SGD momentum/Adam beta1
    opt.weight_decay=0.0005  # optimizer weight decay 5e-4
    opt.warmup_epochs=3.0  # warmup epochs (fractions ok)
    opt.warmup_momentum=0.8  # warmup initial momentum
    opt.warmup_bias_lr=0.1  # warmup initial bias lr
    opt.box=7.5  # box loss gain
    opt.cls=0.5  # cls loss gain (scale with pixels)
    opt.dfl=1.5  # dfl loss gain
    opt.fl_gamma=0.0  # focal loss gamma (efficientDet default gamma=1.5)
    opt.label_smoothing=0.0  # label smoothing (fraction)
    opt.nbs=64  # nominal batch size
    opt.hsv_h=0.015  # image HSV-Hue augmentation (fraction)
    opt.hsv_s=0.7  # image HSV-Saturation augmentation (fraction)
    opt.hsv_v=0.4  # image HSV-Value augmentation (fraction)
    opt.degrees=0.0  # image rotation (+/- deg)
    opt.translate=0.1  # image translation (+/- fraction)
    opt.scale=0.5  # image scale (+/- gain)
    opt.shear=0.0  # image shear (+/- deg)
    opt.perspective=0.0  # image perspective (+/- fraction), range 0-0.001
    opt.flipud=0.0  # image flip up-down (probability)
    opt.fliplr=0.5  # image flip left-right (probability)
    opt.mosaic=1.0  # image mosaic (probability)
    opt.mixup=0.0  # image mixup (probability)
    opt.copy_paste=0.0  # segment copy-paste (probability)
        
    return opt

class Opt():
    pass