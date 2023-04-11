from PIL import Image
import numpy as np
from typing import Tuple, List
import os
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.exceptions import AuthenticationError
from picsellia.types.enums import LogType
import sys
import picsellia
from pycocotools.coco import COCO
import shutil
from picsellia import Client
# from sklearn.metrics import f1_score, recall_score, precision_score



def dataset(experiment, assets, count, split_type, new_size, n_classes):
    X = []
    for i in range(len(assets)):
        path = os.path.join(experiment.png_dir, split_type, assets[i].filename)
        x = Image.open(path).convert('RGB')
        x = x.resize(new_size)
        x = np.array(x)
        X.append(x)

    y = np.zeros(len(assets))
    indices = np.cumsum(count['y'])
    for i in range(len(indices) - 1):
        y[indices[i]:indices[i + 1]] = np.ones(len(y[indices[i]:indices[i + 1]])) * (i)
    y = to_categorical(y, n_classes)

    return np.asarray(X), y


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


def get_train_test_valid_datasets_from_experiment(experiment: Experiment) -> Tuple[DatasetVersion]:
    is_split = _is_train_test_valid_dataset(experiment)
    if is_split:
        print("We found 3 datasets:")
        train: DatasetVersion = experiment.get_dataset('train')
        print(f"{train.name}/{train.version} for training")
        test: DatasetVersion = experiment.get_dataset('test')
        print(f"{test.name}/{test.version} for testing")
        valid: DatasetVersion = experiment.get_dataset('valid')
        print(f"{valid.name}/{valid.version} for validation")
    else:
        print("We only found one dataset inside your experiment, the train/test/split will be performed automatically.")
        train: DatasetVersion = experiment.list_attached_dataset_versions()[0]
        test = None
        valid = None
    return is_split, train, test, valid


def _is_train_test_valid_dataset(experiment: Experiment) -> bool:
    datasets: List[DatasetVersion] = experiment.list_attached_dataset_versions()
    if len(datasets) < 3:
        return False
    template = ["train", "test", "valid"]
    ok_counter = 0
    for dataset in datasets:
        if dataset.version in template:
            ok_counter += 1
    return ok_counter == 3


def _move_files_in_class_directories(coco: COCO, base_imdir: str = None) -> None:
    fnames = os.listdir(base_imdir)
    for i in coco.cats:
        cat = coco.cats[i]
        class_folder = os.path.join(base_imdir, cat["name"])
        if not os.path.isdir(class_folder):
            os.mkdir(class_folder)
    print(f"Formatting {base_imdir} ..")
    for i in coco.imgs:
        im = coco.imgs[i]
        if im["file_name"] not in fnames:
            continue
        ann = coco.loadAnns(im["id"])
        if len(ann) > 1:
            print(f"{im['file_name']} has more than one class. Skipping")
        ann = ann[0]
        cat = coco.loadCats(ann['category_id'])[0]
        fpath = os.path.join(base_imdir, im['file_name'])
        new_fpath = os.path.join(base_imdir, cat['name'], im['file_name'])
        try:
            shutil.move(fpath, new_fpath)
        except Exception as e:
            print(f"{im['file_name']} skipped.")
    print(f"Formatting {base_imdir} .. OK")
    return base_imdir


def recall_m(y_true, y_pred):
    '''
    This function returns recall_score between y_true and y_pred
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns recall score of the batch
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # calculates number of true positives
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))  # calculates number of actual positives
    recall = true_positives / (possible_positives + K.epsilon())  # K.epsilon takes care of non-zero divisions
    return recall


def precision_m(y_true, y_pred):
    '''
    This function returns precison_score between y_true and y_pred
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns prediction score of the batch
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # calculates number of true positives
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  # calculates number of predicted positives
    precision = true_positives / (predicted_positives + K.epsilon())  # K.epsilon takes care of non-zero divisions
    return precision


def f1_micro(y_true, y_pred):
    '''
    This function returns f1_score between y_true and y_pred
    This
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns f1 score of the batch
    '''
    precision = precision_m(y_true, y_pred)  # calls precision metric and takes the score of precision of the batch
    recall = recall_m(y_true, y_pred)  # calls recall metric and takes the score of precision of the batch
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


dependencies = {
    'recall_m': recall_m,
    'precision_m': precision_m,
    'f1_micro': f1_micro
}
