from picsellia_yolov5.yolov5.segment.train import train
from picsellia_yolov5.yolov5.utils.callbacks import Callbacks
from picsellia_yolov5.yolov5.utils.torch_utils import select_device
from picsellia_yolov5.yolov5.utils.general import check_file, check_yaml, increment_path

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
dataset = experiment.list_attached_dataset_versions()[0]

annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
f = open(annotation_path)
annotations_dict = json.load(f)
annotations_coco=COCO(annotation_path)

labelmap = {}
for x in annotations_dict['categories']:
    labelmap[str(x['id'])] = x['name']
experiment.log('labelmap', labelmap, 'labelmap', replace=True)

targetdir = experiment.base_dir

with open(annotation_path, "r") as f:
    annotations = json.load(f)

base_imgdir = experiment.png_dir
prop = 0.7

train_assets, test_assets, train_split, test_split, labels = dataset.train_test_split(prop=prop, random_seed=42)

test_list = test_assets.items.copy()
# test_list = test_assets.items.copy()[:50]
# print(f'TEST_LIST : {test_list}')
random.seed(42)
random.shuffle(test_list)

nb_asset = len(test_list)//2
val_data = test_list[nb_asset:]
test_data = test_list[:nb_asset]
val_assets = MultiAsset(dataset.connexion, dataset.id, val_data)
test_assets = MultiAsset(dataset.connexion, dataset.id, test_data)

train_assets.download(target_path=os.path.join(experiment.png_dir, 'train', 'images'), max_workers=8)
val_assets.download(target_path=os.path.join(experiment.png_dir, 'val', 'images'), max_workers=8)
test_assets.download(target_path=os.path.join(experiment.png_dir, 'test', 'images'), max_workers=8)

picsellia_utils.create_yolo_segmentation_label(experiment, annotations_dict, annotations_coco)
data_yaml_path = picsellia_utils.generate_data_yaml(experiment, labelmap, experiment.base_dir)

experiment.log('train-split', train_split, 'bar', replace=True)
experiment.log('test-split', test_split, 'bar', replace=True)
parameters = experiment.get_log(name='parameters').data

cfg = picsellia_utils.edit_model_yaml(label_map=labelmap, experiment_name=experiment.name, config_path=experiment.config_dir)
opt = picsellia_utils.setup_hyp(
    experiment = experiment,
    data_yaml_path=data_yaml_path,
    config_path=cfg,
    params=parameters,
    label_map=labelmap
    )

# main(opt, callbacks=Callbacks())

opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
if opt.evolve:
    cwd = os.getcwd()
    if opt.project == os.path.join(cwd, 'runs/train'):  # if default project name, rename to runs/evolve
        opt.project = os.path.join(cwd, 'runs/evolve')
    opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
if opt.name == 'cfg':
    opt.name = Path(opt.cfg).stem  # use model.yaml as name
opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
print(f'SAVE_DIR : {opt.save_dir}')

callbacks=Callbacks()
device = select_device(opt.device, batch_size=opt.batch_size)

train(opt.hyp, opt, device, callbacks, pxl=experiment)

picsellia_utils.send_run_to_picsellia(experiment, targetdir)