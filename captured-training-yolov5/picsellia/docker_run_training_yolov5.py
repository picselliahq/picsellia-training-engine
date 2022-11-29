from picsellia import Client
from picsellia_yolov5.utils import to_yolo, find_matching_annotations, edit_model_yaml, generate_yaml, Opt, setup_hyp
from picsellia_yolov5.utils import send_run_to_picsellia
from picsellia_yolov5.yolov5.train import train
import os 
from picsellia.exceptions import AuthenticationError

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

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
    experiment = client.get_experiment_by_id(experiment_id, tree=True, with_artifacts=True)
else:
    if "experiment_name" in os.environ and "project_token" in os.environ:
        project_token = os.environ['project_token']
        experiment_name = os.environ['experiment_name']
        project = client.get_project_by_id(project_token)
        experiment = project.get_experiment(experiment_name, tree=True, with_artifacts=True)
    else:
        raise AuthenticationError("You must either set the experiment id or the project token + experiment_name")

experiment.download_annotations()
experiment.download_pictures()
experiment.generate_labelmap()
experiment.log('labelmap', experiment.label_map, 'labelmap', replace=True)

YOLODIR = 'YOLO-{}'.format(experiment_name)
train_set, test_set = to_yolo(pxl_annotations_dict=experiment.dict_annotations,labelmap=experiment.label_map, base_imgdir=experiment.png_dir, targetdir=YOLODIR, prop=0.7, copy_image=False) 

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

generate_yaml(yamlname=experiment_name, targetdir=YOLODIR, labelmap=experiment.label_map)
cfg = edit_model_yaml(label_map=experiment.label_map, experiment_name=experiment_name, config_path=experiment.config_dir)
hyp, opt, device = setup_hyp(experiment_name, cfg, experiment.checkpoint_dir, experiment.get_log('parameters'), experiment.label_map)
train(hyp, opt, opt.device, pxl=experiment)
send_run_to_picsellia(experiment, YOLODIR)
experiment.update(statut="success")