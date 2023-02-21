from picsellia_yolov5.yolov5.segment.train import train
from picsellia_yolov5.yolov5.utils.callbacks import Callbacks
from picsellia_yolov5.yolov5.utils.torch_utils import select_device
from picsellia_yolov5 import picsellia_utils

from picsellia.types.enums import AnnotationFileType
from picsellia.sdk.asset import MultiAsset

import random
import json 
import os 
import logging
from pathlib import Path

from pycocotools.coco import COCO

os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
# os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

logging.getLogger('picsellia').setLevel(logging.INFO)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))

experiment = picsellia_utils.get_experiment()
experiment.download_artifacts(with_tree=True)
current_dir = os.path.join(os.getcwd(), experiment.base_dir)
base_imgdir = experiment.png_dir

parameters = experiment.get_log(name='parameters').data

if len(experiment.list_attached_dataset_versions())==3:
    train_ds, val_ds, test_ds = experiment.get_dataset(name='train'), experiment.get_dataset(name='val'), experiment.get_dataset(name='test')
    
    for data_type, dataset in {'train' : train_ds, 'val' : val_ds, 'test' : test_ds}.items():
        annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, current_dir)
        f = open(annotation_path)
        annotations_dict = json.load(f)
        annotations_coco=COCO(annotation_path)
        
        if data_type=='train':
            labelmap = {}
            for x in annotations_dict['categories']:
                labelmap[str(x['id'])] = x['name']
        
        dataset.list_assets().download(target_path=os.path.join(base_imgdir, data_type, 'images'), max_workers=8)
        picsellia_utils.create_yolo_segmentation_label(experiment, data_type, annotations_dict, annotations_coco)
    
else: 
    dataset = experiment.list_attached_dataset_versions()[0]
    
    annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, current_dir)
    f = open(annotation_path)
    annotations_dict = json.load(f)
    annotations_coco=COCO(annotation_path)
    labelmap = {}
    for x in annotations_dict['categories']:
        labelmap[str(x['id'])] = x['name']
    
    prop = 0.7 if not 'prop_train_split' in parameters.keys() else parameters["prop_train_split"]

    train_assets, test_assets, val_assets = picsellia_utils.train_test_val_split(experiment, dataset, prop)
    
    for data_type, assets in {'train' : train_assets, 'val' : val_assets, 'test' : test_assets}.items():
        assets.download(target_path=os.path.join(base_imgdir, data_type, 'images'), max_workers=8)
        picsellia_utils.create_yolo_segmentation_label(experiment, data_type, annotations_dict, annotations_coco)

experiment.log('labelmap', labelmap, 'labelmap', replace=True)
cwd = os.getcwd()
data_yaml_path = picsellia_utils.generate_data_yaml(experiment, labelmap, current_dir)
cfg = picsellia_utils.edit_model_yaml(label_map=labelmap, experiment_name=experiment.name, config_path=experiment.config_dir)
opt = picsellia_utils.setup_hyp(
    experiment = experiment,
    data_yaml_path=data_yaml_path,
    config_path=cfg,
    params=parameters,
    label_map=labelmap,
    cwd=cwd
    )

picsellia_utils.check_files(opt)

callbacks=Callbacks()
device = select_device(opt.device, batch_size=opt.batch_size)

train(opt.hyp, opt, device, callbacks, pxl=experiment)

picsellia_utils.send_run_to_picsellia(experiment, cwd)