from picsellia import Client
from picsellia_yolov5.yolov5.train import train
from picsellia_yolov5.yolov5.models.export import export
from picsellia.exceptions import AuthenticationError
from picsellia.types.enums import AnnotationFileType, LogType
from picsellia_yolov5 import utils
import json 
import os 


os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 

os.chdir('picsellia')
if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")

api_token = os.environ['api_token']

if "host" not in os.environ:
    host = "https://app.picsellia.com/sdk/v1"
else:
    host = os.environ["host"]

client = Client(
    api_token=api_token,
    host=host
)

if "experiment_id" in os.environ:
    experiment_id = os.environ['experiment_id']
    experiment = client.get_experiment_by_id(experiment_id)
    experiment.download_artifacts(with_tree=True)

else:
    if "experiment_name" in os.environ and "project_token" in os.environ:
        project_token = os.environ['project_token']
        experiment_name = os.environ['experiment_name']
        project = client.get_project_by_id(project_token)
        experiment = project.get_experiment(experiment_name)
        experiment.download_artifacts(with_tree=True)
    else:
        raise AuthenticationError("You must either set the experiment id or the project token + experiment_name")

dataset = experiment.list_attached_dataset_versions()[0]
dataset.download(experiment.png_dir)

annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
labels = dataset.list_labels()
labelmap = {}
for i, label in enumerate(labels):
    labelmap[str(i+1)] = label.name
experiment.log('labelmap', labelmap, LogType.TABLE, replace=True)

targetdir = 'YOLO-{}'.format(experiment_name)


with open(annotation_path, "r") as f:
    annotations = json.load(f)

base_imgdir = experiment.png_dir
prop = 0.7

train_assets, test_assets, train_split, test_split, labels = dataset.train_test_split(prop=prop)
utils.to_yolo(
    assets=train_assets, 
    annotations=annotations, 
    base_imgdir=base_imgdir,
    targetdir=targetdir, 
    copy_image=True, 
    split="train"
)
utils.to_yolo(
    assets=test_assets, 
    annotations=annotations, 
    base_imgdir=base_imgdir,
    targetdir=targetdir, 
    copy_image=True, 
    split="test"
)


experiment.log('train-split', train_split, 'bar', replace=True)
experiment.log('test-split', test_split, 'bar', replace=True)
parameters = experiment.get_log(name='parameters').data


data_yaml_path = utils.generate_yaml(
    yamlname=experiment.name,
    datatargetdir=experiment.base_dir, 
    imgdir=targetdir, 
    labelmap=labelmap
    )
cfg = utils.edit_model_yaml(label_map=labelmap, experiment_name=experiment.name, config_path=experiment.config_dir)
hyp, opt, device = utils.setup_hyp(
    experiment = experiment,
    data_yaml_path=data_yaml_path,
    config_path=cfg,
    params=parameters,
    label_map=labelmap
    )

model = train(hyp, opt, opt.device, pxl=experiment)

utils.send_run_to_picsellia(experiment, targetdir)