import os
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
import logging
import json
from classification_models.keras import Classifiers
from picsellia.types.enums import AnnotationFileType
import tensorflow as tf
from picsellia.types.enums import LogType, ExperimentStatus
import keras
from picsellia.exceptions import AuthenticationError
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger('picsellia').setLevel(logging.INFO)

if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")

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
    raise AuthenticationError("You must set the project_token or project_name and experiment_name")

experiment.start_logging_chapter('Dowloading files')

experiment.download_artifacts(with_tree=True)

artifact = experiment.get_artifact('keras-model')
model_name = artifact.filename
model_path = os.path.join(experiment.checkpoint_dir, model_name)
os.rename(os.path.join(experiment.base_dir, model_name), model_path)

dataset = experiment.list_attached_dataset_versions()[0]

annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
f = open(annotation_path)
annotations_dict = json.load(f)

train_assets, eval_assets, count_train, count_eval, labels = dataset.train_test_split()

labelmap = {}
for x in annotations_dict['categories']:
    labelmap[str(x['id'])] = x['name']

experiment.log('labelmap', labelmap, 'labelmap', replace=True)
experiment.log('train-split', count_train, 'bar', replace=True)
experiment.log('test-split', count_eval, 'bar', replace=True)
parameters = experiment.get_log(name='parameters').data

train_assets.download(target_path=os.path.join(experiment.png_dir, 'train'))
eval_assets.download(target_path=os.path.join(experiment.png_dir, 'eval'))

n_classes = len(labelmap)

experiment.start_logging_chapter('Init Model')