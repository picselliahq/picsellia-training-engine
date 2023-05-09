
import os
import random
import re
import shutil
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import tqdm
import yaml
from picsellia import Client
from picsellia.types.enums import LogType
from picsellia.sdk.asset import Asset, MultiAsset
from picsellia.sdk.dataset import Dataset
from picsellia.exceptions import NoDataError

def find_image_id(annotations, fname):
    for image in annotations["images"]:
        if image["file_name"] == fname:
            return image["id"]
    return None

def find_matching_annotations(dict_annotations=None, fname=None):
    img_id = find_image_id(dict_annotations, fname=fname)
    if img_id is None:
        return False, None
    ann_array = []
    for ann in dict_annotations["annotations"]:
        if ann["image_id"] == img_id:
            ann_array.append(ann)
    return True, ann_array

def to_yolo(assets=None, annotations=None, base_imgdir=None, targetdir=None,copy_image=True, split="train"):
    """
        Simple utility function to transcribe a Picsellia Format Dataset into YOLOvX
    """
    step = split
    # Creating tree directory for YOLO
    if not os.path.isdir(targetdir):
        os.mkdir(targetdir)

    for dirname in ["images", "labels"]:
        if not os.path.isdir(os.path.join(targetdir, dirname)):
            os.mkdir(os.path.join(targetdir, dirname))

    for path in os.listdir(targetdir):
        if not os.path.isdir(os.path.join(targetdir, path, step)):
            os.mkdir(os.path.join(targetdir, path, step))

    for asset in tqdm.tqdm(assets):
        width, height = asset.width, asset.height
        success, objs = find_matching_annotations(annotations, asset.filename)

        if copy_image:
            shutil.copy(os.path.join(base_imgdir, asset.filename), os.path.join(targetdir,"images", step, asset.filename,))
        else:
            shutil.move(os.path.join(base_imgdir, asset.filename), os.path.join(targetdir, 'images', step, asset.filename))

        if success:
            label_name = "{}.txt".format(os.path.splitext(asset.filename)[0])
            with open(os.path.join(targetdir,'labels',step, label_name), 'w') as f:
                for a in objs:
                    x1, y1, w, h = a["bbox"]
                    category_id = a["category_id"]
                    f.write(f"{category_id} {(x1 + w / 2)/width} {(y1 + h / 2)/height} {w/width} {h/height}\n")  
        else:
            continue
    return 
        

def generate_yaml(yamlname, datatargetdir, imgdir,  labelmap):
    if not os.path.isdir(os.path.join(datatargetdir, "data")):
        os.mkdir(os.path.join(datatargetdir, "data"))

    dict_file = {   
                'train' : '{}/{}/train'.format(imgdir, "images"),
                'val' : '{}/{}/test'.format(imgdir, "images"),
                'nc': len(labelmap),
                'names': list(labelmap.values())
            }
    
    opath = '{}/data/{}.yaml'.format(datatargetdir, yamlname)
    with open(opath, 'w') as file:
        yaml.dump(dict_file, file)
    return opath

def generate_data_yaml(exp, labelmap, config_path):
    
    cwd = os.getcwd()
    
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    data_config_path = os.path.join(config_path, 'data_config.yaml')
    n_classes = len(labelmap)
    labelmap = {int(k): v for k, v in labelmap.items()}
    ordered_labelmap = dict(sorted(OrderedDict(labelmap).items()))
    data_config = {
        'train' : os.path.join(cwd, exp.png_dir, 'train'),
        'val' : os.path.join(cwd, exp.png_dir, 'val'),
        'test' : os.path.join(cwd, exp.png_dir, 'test'),
        'nc' : n_classes,
        'names' : list(ordered_labelmap.values())
    }
    with open(data_config_path, 'w+') as f:
        yaml.dump(data_config, f, allow_unicode=True)
    return data_config_path

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
            weight_path = os.path.join(cwd, experiment.checkpoint_dir, f)
        if f.endswith('.yaml'):
            hyp_path = os.path.join(cwd, experiment.checkpoint_dir, f)
    
    opt = Opt()
    
    opt.task = task
    opt.mode = 'train'
    opt.cwd = cwd
    # Train settings -------------------------------------------------------------------------------------------------------
    opt.model = weight_path 
    opt.data = data_yaml_path
    opt.epochs = 100 if not 'epochs' in params.keys() else params["epochs"]
    opt.patience = 5
    opt.batch = 4 if not 'batch_size' in params.keys() else params["batch_size"]
    opt.imgsz = 640 if not 'input_shape' in params.keys() else params["input_shape"]
    opt.save = True
    opt.save_period = 100 if not 'save_period' in params.keys() else params["save_period"]
    opt.cache = False
    opt.device = '0' if torch.cuda.is_available() else 'cpu'
    opt.workers = 8
    opt.project = os.path.join(cwd, 'runs/train')
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

def sort_split(split_dict, labels_names):
    label_num_list = [(split_dict['x'][i], split_dict['y'][i]) for i in range(len(split_dict['x']))]
    for label in labels_names:
        if label not in split_dict['x']:
            label_num_list.append((label, 0))
    label_num_dict = dict(label_num_list)
    sorted_split_dict = {
        'x' : labels_names,
        'y' : [label_num_dict[label] for label in labels_names]
    }
    return sorted_split_dict

def train_test_val_split(experiment, dataset, prop, dataset_length, label_names):
    train_assets, test_assets, train_split, test_split, labels = picsellia_train_test_split(dataset, prop=prop, random_seed=42, dataset_length=dataset_length)
    
    sorted_train_split = sort_split(train_split, label_names)
    sorted_test_split = sort_split(test_split, label_names)
    
    experiment.log('train-split', sorted_train_split, 'bar', replace=True)
    experiment.log('test-split', sorted_test_split, 'bar', replace=True)

    test_list = test_assets.items.copy()
    random.seed(42)
    random.shuffle(test_list)

    nb_asset = len(test_list)//2
    val_data = test_list[nb_asset:]
    test_data = test_list[:nb_asset]
    val_assets = MultiAsset(dataset.connexion, dataset.id, val_data)
    test_assets = MultiAsset(dataset.connexion, dataset.id, test_data)
    return train_assets, test_assets, val_assets

def picsellia_train_test_split(
    dataset: Dataset,
    prop: float = 0.8,
    random_seed=None,
    dataset_length: int = 0,
):
    nb_pages = int(dataset_length / 100) + 1
    extended_assets = {"items": []}
    for page in range(1, nb_pages + 1):
        params = {"limit": 100, "offset": (page - 1) * 100}
        try:
            r = dataset.connexion.get(
                f"/sdk/dataset/version/{dataset.id}/assets/extended", params=params
            ).json()
            if r["count"] == 0:
                raise NoDataError("No asset with annotation found in this dataset")
        except Exception:
            pass
        extended_assets["items"] = extended_assets["items"] + r["items"]

    count = 0
    items = []
    for item in extended_assets["items"]:
        if not item["annotations"]:
            continue

        count += 1
        items.append(item)

    if random_seed is not None:
        random.seed(random_seed)

    nb_assets_train = int(count * prop)
    train_eval_rep = [1] * nb_assets_train + [0] * (count - nb_assets_train)
    random.shuffle(train_eval_rep)

    labels = dataset.list_labels()
    label_names = {str(label.id): label.name for label in labels}

    k = 0

    train_assets = []
    eval_assets = []

    train_label_count = {}
    eval_label_count = {}
    for item in items:
        annotations = item["annotations"]

        # TODO: Get only from worker or status
        annotation = annotations[0]

        asset = Asset(dataset.connexion, dataset_version_id=dataset.id, data=item)

        if train_eval_rep[k] == 0:
            eval_assets.append(asset)
            label_count_ref = eval_label_count
        else:
            train_assets.append(asset)
            label_count_ref = train_label_count

        k += 1

        label_ids = []
        for shape in annotation["rectangles"]:
            label_ids.append(shape["label_id"])

        for shape in annotation["classifications"]:
            label_ids.append(shape["label_id"])

        for shape in annotation["points"]:
            label_ids.append(shape["label_id"])

        for shape in annotation["polygons"]:
            label_ids.append(shape["label_id"])

        for shape in annotation["lines"]:
            label_ids.append(shape["label_id"])

        for label_id in label_ids:
            try:
                label_name = label_names[label_id]
                if label_name not in label_count_ref:
                    label_count_ref[label_name] = 1
                else:
                    label_count_ref[label_name] += 1
            except KeyError:  # pragma: no cover
                pass

    train_repartition = {
        "x": list(train_label_count.keys()),
        "y": list(train_label_count.values()),
    }

    eval_repartition = {
        "x": list(eval_label_count.keys()),
        "y": list(eval_label_count.values()),
    }

    return (
        MultiAsset(dataset.connexion, dataset.id, train_assets),
        MultiAsset(dataset.connexion, dataset.id, eval_assets),
        train_repartition,
        eval_repartition,
        labels,
    )
    

def create_yolo_detection_label(exp, data_type, annotations_dict, annotations_coco, label_names):
    
    dataset_path = os.path.join(exp.png_dir, data_type)
    image_filenames = os.listdir(os.path.join(dataset_path, 'images'))

    labels_path = os.path.join(dataset_path, 'labels')

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for img in annotations_dict['images']:
        img_filename = img['file_name']
        if img_filename in image_filenames :
            create_img_label_detection(img, annotations_coco, labels_path, label_names)

def create_img_label_detection(img, annotations_coco, labels_path, label_names):
    result = []
    img_id = img['id']
    img_filename = img['file_name']
    w = img['width']
    h = img['height']
    txt_name = os.path.splitext(img_filename)[0] + '.txt'
    annotation_ids = annotations_coco.getAnnIds(imgIds=img_id)
    anns = annotations_coco.loadAnns(annotation_ids)
    for ann in anns:
        bbox = ann['bbox']
        yolo_bbox = coco_to_yolo_detection(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        seg_string = " ".join([str(x) for x in yolo_bbox])
        label = label_names.index(annotations_coco.loadCats(ann['category_id'])[0]['name'])
        result.append(f"{label} {seg_string}")
    with open(os.path.join(labels_path, txt_name), 'w') as f:
        f.write("\n".join(result))
                    
def coco_to_yolo_detection(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]


def create_yolo_segmentation_label(exp, data_type, annotations_dict, annotations_coco):
    
    dataset_path = os.path.join(exp.png_dir, data_type)
    image_filenames = os.listdir(os.path.join(dataset_path, 'images'))

    labels_path = os.path.join(dataset_path, 'labels')

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for img in annotations_dict['images']:
        img_filename = img['file_name']
        if img_filename in image_filenames :
            create_img_label_segmentation(img, annotations_coco, labels_path)

def create_img_label_segmentation(img, annotations_coco, labels_path):
    result = []
    img_id = img['id']
    img_filename = img['file_name']
    w = img['width']
    h = img['height']
    txt_name = os.path.splitext(img_filename)[0] + '.txt'
    annotation_ids = annotations_coco.getAnnIds(imgIds=img_id)
    anns = annotations_coco.loadAnns(annotation_ids)
    for ann in anns:
        seg = coco_to_yolo_segmentation(ann['segmentation'], w, h)
        seg_string = " ".join([str(x) for x in seg])
        result.append(f"{ann['category_id']} {seg_string}")
    with open(os.path.join(labels_path, txt_name), 'w') as f:
        f.write("\n".join(result))
        
def countList(lst1, lst2):
    return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]

def coco_to_yolo_segmentation(ann, image_w, image_h):
    pair_index = np.arange(0, len(ann[0]), 2)
    impair_index = np.arange(1, len(ann[0]), 2)
    Xs = list(map(ann[0].__getitem__, pair_index))
    xs = list(map(lambda x: x/image_w, Xs))
    Ys = list(map(ann[0].__getitem__, impair_index))
    ys = list(map(lambda x: x/image_h, Ys))
    return countList(xs, ys)

def find_final_run(cwd):
    runs_path = os.path.join(cwd, 'runs', 'train')
    dirs = os.listdir(runs_path)
    dirs.sort()
    if len(dirs) == 1:
        return os.path.join(runs_path, dirs[0])
    base = dirs[0][:7]
    truncate_dirs = [n[len(base)-1:] for n in dirs]
    last_run_nb = max(truncate_dirs)[-1]
    if last_run_nb=='p':
        last_run_nb=''
    return os.path.join(runs_path, base + last_run_nb)

def get_batch_mosaics(final_run_path):
    val_batch0_labels = None
    val_batch0_pred = None
    val_batch1_labels = None
    val_batch1_pred = None
    val_batch2_labels = None
    val_batch2_pred = None
    if os.path.isfile(os.path.join(final_run_path, 'val_batch0_labels.jpg')):
        val_batch0_labels = os.path.join(final_run_path, 'val_batch0_labels.jpg')
    if os.path.isfile(os.path.join(final_run_path, 'val_batch0_pred.jpg')):
        val_batch0_pred = os.path.join(final_run_path, 'val_batch0_pred.jpg')
    if os.path.isfile(os.path.join(final_run_path, 'val_batch1_labels.jpg')):
        val_batch1_labels = os.path.join(final_run_path, 'val_batch1_labels.jpg')
    if os.path.isfile(os.path.join(final_run_path, 'val_batch1_pred.jpg')):
        val_batch1_pred = os.path.join(final_run_path, 'val_batch1_pred.jpg')
    if os.path.isfile(os.path.join(final_run_path, 'val_batch2_labels.jpg')):
        val_batch2_labels = os.path.join(final_run_path, 'val_batch2_labels.jpg')
    if os.path.isfile(os.path.join(final_run_path, 'val_batch2_pred.jpg')):
        val_batch2_pred = os.path.join(final_run_path, 'val_batch2_pred.jpg')
    return val_batch0_labels, val_batch0_pred, val_batch1_labels, val_batch1_pred, val_batch2_labels, val_batch2_pred

def get_weights_and_config(final_run_path):
    best_weights = None
    hyp_yaml = None
    if os.path.isfile(os.path.join(final_run_path, 'weights', 'best.pt')):
        best_weights = os.path.join(final_run_path, 'weights', 'best.pt')
    if os.path.isfile(os.path.join(final_run_path, 'hyp.yaml')):
        hyp_yaml = os.path.join(final_run_path, 'hyp.yaml')
    if os.path.isfile(os.path.join(final_run_path, 'args.yaml')):
        hyp_yaml = os.path.join(final_run_path, 'args.yaml')
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
    if os.path.isfile(os.path.join(final_run_path, 'confusion_matrix.png')):
        confusion_matrix = os.path.join(final_run_path, 'confusion_matrix.png')
    if os.path.isfile(os.path.join(final_run_path, 'F1_curve.png')):
        F1_curve = os.path.join(final_run_path, 'F1_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'labels_correlogram.jpg')):
        labels_correlogram = os.path.join(final_run_path, 'labels_correlogram.jpg')
    if os.path.isfile(os.path.join(final_run_path, 'labels.jpg')):
        labels = os.path.join(final_run_path, 'labels.jpg')
    if os.path.isfile(os.path.join(final_run_path, 'P_curve.png')):
        P_curve = os.path.join(final_run_path, 'P_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'PR_curve.png')):
        PR_curve = os.path.join(final_run_path, 'PR_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'R_curve.png')):
        R_curve = os.path.join(final_run_path, 'R_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'BoxF1_curve.png')):
        BoxF1_curve = os.path.join(final_run_path, 'BoxF1_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'BoxP_curve.png')):
        BoxP_curve = os.path.join(final_run_path, 'BoxP_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'BoxPR_curve.png')):
        BoxPR_curve = os.path.join(final_run_path, 'BoxPR_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'BoxR_curve.png')):
        BoxR_curve = os.path.join(final_run_path, 'BoxR_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'MaskF1_curve.png')):
        MaskF1_curve = os.path.join(final_run_path, 'MaskF1_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'MaskP_curve.png')):
        MaskP_curve = os.path.join(final_run_path, 'MaskP_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'MaskPR_curve.png')):
        MaskPR_curve = os.path.join(final_run_path, 'MaskPR_curve.png')
    if os.path.isfile(os.path.join(final_run_path, 'MaskR_curve.png')):
        MaskR_curve = os.path.join(final_run_path, 'MaskR_curve.png')
    return confusion_matrix, F1_curve, labels_correlogram, labels, P_curve, PR_curve, R_curve, BoxF1_curve, BoxP_curve, BoxPR_curve, BoxR_curve, MaskF1_curve, MaskP_curve, MaskPR_curve, MaskR_curve

def send_run_to_picsellia(experiment, cwd, save_dir=None):
    if save_dir is not None:
        final_run_path=save_dir
    else:
        final_run_path = find_final_run(cwd)
    best_weigths, hyp_yaml = get_weights_and_config(final_run_path)
    
    model_latest_path = os.path.join(final_run_path, 'weights', 'best.onnx')

    if model_latest_path is not None:
        experiment.store('model-latest', model_latest_path)
    if best_weigths is not None:
        experiment.store('checkpoint-index-latest', best_weigths)
    if hyp_yaml is not None:
        experiment.store('checkpoint-data-latest', hyp_yaml)
    for curve in get_metrics_curves(final_run_path):
        if curve is not None:
            name = curve.split('/')[-1].split('.')[0]
            experiment.log(name, curve, LogType.IMAGE)
    for batch in get_batch_mosaics(final_run_path):
        if batch is not None:
            name = batch.split('/')[-1].split('.')[0]
            experiment.log(name, batch, LogType.IMAGE)

def get_experiment():
    if 'api_token' not in os.environ:
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

    client = Client(
        api_token=api_token,
        host=host,
        organization_id=organization_id
    )

    if "experiment_name" in os.environ:
        experiment_name = os.environ["experiment_name"]
        if "project_token" in os.environ:
            project_token = os.environ["project_token"]
            project = client.get_project_by_id(project_token)
        elif "project_name" in os.environ:
            project_name = os.environ["project_name"]
            project = client.get_project(project_name)
        experiment = project.get_experiment(experiment_name)
    else:
        raise Exception("You must set the project_token or project_name and experiment_name")
    return experiment