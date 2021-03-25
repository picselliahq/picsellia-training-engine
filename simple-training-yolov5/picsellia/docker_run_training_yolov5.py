from picsellia.client import Client
from picsellia_yolov5.utils import to_yolo, find_matching_annotations, edit_model_yaml, generate_yaml, Opt, setup_hyp
from picsellia_yolov5.utils import send_run_to_picsellia
from picsellia_yolov5.yolov5.train import train
import argparse 
import sys
import os 
import subprocess 
import yaml 
import torch

os.chdir('picsellia')
if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")

api_token = os.environ['api_token']

if "experiment_id" in os.environ:
    experiment_id = os.environ['experiment_id']

    experiment = Client.Experiment(api_token=api_token)
    exp = experiment.checkout(experiment_id, tree=True, with_file=True)
else:
    if "experiment_name" in os.environ and "project_token" in os.environ:
        project_token = os.environ['project_token']
        experiment_name = os.environ['experiment_name']
        experiment = Client.Experiment(api_token=api_token, project_token=project_token)
        exp = experiment.checkout(experiment_name, tree=True, with_file=True)
    else:
        raise AuthenticationError("You must either set the experiment id or the project token + experiment_name")

experiment.dl_annotations()
experiment.dl_pictures()
experiment.generate_labelmap()
experiment.log('labelmap', experiment.label_map, 'labelmap', replace=True)

YOLODIR = 'YOLO-{}'.format(experiment_name)
train_set, test_set = to_yolo(pxl_annotations_dict=exp.dict_annotations,labelmap=exp.label_map, base_imgdir=exp.png_dir, targetdir=YOLODIR, prop=0.7, copy_image=False) 

train_split = {
    'x': train_set["categories"],
    'y': train_set["train_repartition"],
    'image_list': train_set["image_list"],
}
experiment.log('train-split', train_split, 'bar', replace=True)

test_split = {
    'x': test_set["categories"],
    'y': test_set["train_repartition"],
    'image_list': test_set["image_list"],
}
experiment.log('test-split', test_split, 'bar', replace=True)

generate_yaml(yamlname=experiment_name, targetdir=YOLODIR, labelmap=exp.label_map)
cfg = edit_model_yaml(label_map=exp.label_map, experiment_name=experiment_name, config_path=exp.config_dir)
hyp, opt, device = setup_hyp(experiment_name, cfg, exp.checkpoint_dir, exp.get_data('parameters'), exp.label_map)

train(hyp, opt, opt.device, pxl=exp)

send_run_to_picsellia(exp, YOLODIR)