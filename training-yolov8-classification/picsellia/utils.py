from PIL import Image
import numpy as np
import os
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.dataset_version import DatasetVersion

from pycocotools.coco import COCO
import shutil
from picsellia.types.enums import AnnotationFileType
from picsellia.exceptions import ResourceNotFoundError

# from sklearn.metrics import f1_score, recall_score, precision_score


def dataset(experiment, assets, count, split_type, new_size, n_classes):
    X = []
    for i in range(len(assets)):
        path = os.path.join(experiment.png_dir, split_type, assets[i].filename)
        x = Image.open(path).convert("RGB")
        x = x.resize(new_size)
        x = np.array(x)
        X.append(x)

    y = np.zeros(len(assets))
    indices = np.cumsum(count["y"])
    for i in range(len(indices) - 1):
        y[indices[i] : indices[i + 1]] = np.ones(
            len(y[indices[i] : indices[i + 1]])
        ) * (i)
    y = to_categorical(y, n_classes)

    return np.asarray(X), y


def recall_m(y_true, y_pred):
    """
    This function returns recall_score between y_true and y_pred
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns recall score of the batch
    """
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1))
    )  # calculates number of true positives
    possible_positives = K.sum(
        K.round(K.clip(y_true, 0, 1))
    )  # calculates number of actual positives
    recall = true_positives / (
        possible_positives + K.epsilon()
    )  # K.epsilon takes care of non-zero divisions
    return recall


def precision_m(y_true, y_pred):
    """
    This function returns precison_score between y_true and y_pred
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns prediction score of the batch
    """
    true_positives = K.sum(
        K.round(K.clip(y_true * y_pred, 0, 1))
    )  # calculates number of true positives
    predicted_positives = K.sum(
        K.round(K.clip(y_pred, 0, 1))
    )  # calculates number of predicted positives
    precision = true_positives / (
        predicted_positives + K.epsilon()
    )  # K.epsilon takes care of non-zero divisions
    return precision


def f1_micro(y_true, y_pred):
    """
    This function returns f1_score between y_true and y_pred
    This
    This function is ported as a metric to the Neural Network Models
    Keras backend is used to take care of batch type training, the metric takes in a batch of y_pred and corresponding y_pred
    as input and returns f1 score of the batch
    """
    precision = precision_m(
        y_true, y_pred
    )  # calls precision metric and takes the score of precision of the batch
    recall = recall_m(
        y_true, y_pred
    )  # calls recall metric and takes the score of precision of the batch
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


dependencies = {"recall_m": recall_m, "precision_m": precision_m, "f1_micro": f1_micro}


def create_and_log_labelmap(experiment: Experiment) -> dict:
    names = os.listdir("data/train")  # class names list
    labelmap = {str(i): label for i, label in enumerate(sorted(names))}
    experiment.log("labelmap", labelmap, "labelmap", replace=True)
    return labelmap


def prepare_datasets_with_annotation(
    train_set: DatasetVersion, test_set: DatasetVersion, val_set: DatasetVersion
):
    coco_train, coco_test, coco_val = _create_coco_objects(train_set, test_set, val_set)

    _move_files_in_class_directories(coco_train, "data/train")
    _move_files_in_class_directories(coco_test, "data/test")
    _move_files_in_class_directories(coco_val, "data/val")

    evaluation_ds = test_set
    evaluation_assets = evaluation_ds.list_assets()

    return evaluation_ds, evaluation_assets


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
        ann = coco.loadAnns(coco.getAnnIds(im["id"]))
        if len(ann) > 1:
            print(f"{im['file_name']} has more than one class. Skipping")
        ann = ann[0]
        cat = coco.loadCats(ann["category_id"])[0]
        fpath = os.path.join(base_imdir, im["file_name"])
        new_fpath = os.path.join(base_imdir, cat["name"], im["file_name"])
        try:
            shutil.move(fpath, new_fpath)
            pass
        except Exception as e:
            print(f"{im['file_name']} skipped.")
    print(f"Formatting {base_imdir} .. OK")
    return base_imdir


def _create_coco_objects(
    train_set: DatasetVersion, test_set: DatasetVersion, val_set: DatasetVersion
):
    train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)

    test_annotation_path = test_set.export_annotation_file(AnnotationFileType.COCO)
    coco_test = COCO(test_annotation_path)

    val_annotation_path = val_set.export_annotation_file(AnnotationFileType.COCO)
    coco_val = COCO(val_annotation_path)

    return coco_train, coco_test, coco_val


def _get_three_attached_datasets(
    experiment: Experiment,
) -> tuple[DatasetVersion, DatasetVersion, DatasetVersion]:
    try:
        train_set = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', 'eval')"
        )
    try:
        test_set = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                            expecting 'train', 'test', 'eval')"
        )
    try:
        eval_set = experiment.get_dataset(name="val")
    except Exception:
        try:
            eval_set = experiment.get_dataset(name="eval")
        except Exception:
            raise ResourceNotFoundError(
                "Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                expecting 'train', 'test', 'eval')"
            )
    return train_set, test_set, eval_set


def _transform_two_attached_datasets_to_three(
    experiment: Experiment,
) -> tuple[DatasetVersion, DatasetVersion, DatasetVersion]:
    try:
        train_set = experiment.get_dataset("train")
        test_set = experiment.get_dataset("test")
        eval_set = experiment.get_dataset("test")
    except Exception:
        raise ResourceNotFoundError(
            "Found 2 attached datasets, expecting 'train' and 'test' "
        )
    return train_set, test_set, eval_set


def get_train_test_eval_datasets_from_experiment(
    experiment: Experiment,
) -> tuple[bool, bool, DatasetVersion, DatasetVersion, DatasetVersion]:
    number_of_attached_datasets = len(experiment.list_attached_dataset_versions())
    is_split_three, is_split_two = False, False
    if number_of_attached_datasets == 3:
        is_split_three = True
        train_set, test_set, eval_set = _get_three_attached_datasets(experiment)
    elif number_of_attached_datasets == 2:
        is_split_two = True
        train_set, test_set, eval_set = _transform_two_attached_datasets_to_three(
            experiment
        )
    elif number_of_attached_datasets == 1:
        print(
            "We only found one dataset inside your experiment, the train/test/split will be performed automatically."
        )
        train_set: DatasetVersion = experiment.list_attached_dataset_versions()[0]
        test_set = None
        eval_set = None

    else:
        print("We need at least 1 and at most 3 datasets attached to this experiment ")

    return is_split_two, is_split_three, train_set, test_set, eval_set


def split_single_dataset(experiment: Experiment, train_set: DatasetVersion):
    parameters = experiment.get_log("parameters").data
    prop = (
        0.7
        if not "prop_train_split" in parameters.keys()
        else parameters["prop_train_split"]
    )
    (
        train_assets,
        test_assets,
        eval_assets,
        train_rep,
        test_rep,
        val_rep,
        labels,
    ) = train_set.train_test_val_split([prop, (1 - prop) / 2, (1 - prop) / 2])

    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    for asset in train_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join("data/train", asset.filename)
        shutil.move(old_path, new_path)

    for asset in test_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join("data/test", asset.filename)
        shutil.move(old_path, new_path)

    for asset in eval_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join("data/val", asset.filename)
        shutil.move(old_path, new_path)

    return train_assets, test_assets, eval_assets, train_rep, test_rep, val_rep, labels


def _move_all_files_in_class_directories(train_set: DatasetVersion):
    train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)
    _move_files_in_class_directories(coco_train, "data/train")
    _move_files_in_class_directories(coco_train, "data/test")
    _move_files_in_class_directories(coco_train, "data/val")


def download_triple_dataset(train_set, test_set, eval_set):
    for data_type, dataset in {
        "train": train_set,
        "test": test_set,
        "val": eval_set,
    }.items():
        dataset.download(target_path=os.path.join("data", data_type), max_workers=8)


def log_split_dataset_repartition_to_experiment(
    experiment: Experiment, train_rep, test_rep, val_rep
) -> dict:
    names = os.listdir("data/train")  # class names list
    labelmap = {str(i): label for i, label in enumerate(sorted(names))}
    experiment.log(
        "train-split",
        order_repartition_according_labelmap(labelmap, train_rep),
        "bar",
        replace=True,
    )
    experiment.log(
        "test-split",
        order_repartition_according_labelmap(labelmap, test_rep),
        "bar",
        replace=True,
    )
    experiment.log(
        "val-split",
        order_repartition_according_labelmap(labelmap, val_rep),
        "bar",
        replace=True,
    )
    return labelmap


def order_repartition_according_labelmap(labelmap, repartition):
    ordered_rep = {"x": list(labelmap.values()), "y": []}
    for name in ordered_rep["x"]:
        ordered_rep["y"].append(repartition["y"][repartition["x"].index(name)])
    return ordered_rep


def log_labelmap(experiment: Experiment):
    names = os.listdir("data/train")  # class names list
    labelmap = {str(i): label for i, label in enumerate(sorted(names))}
    experiment.log("labelmap", labelmap, "labelmap", replace=True)


def predict_class(labelmap: dict, val_folder_path: str, model):
    gt_class = []
    pred_class = []
    for class_id, label in labelmap.items():
        label_path = os.path.join(val_folder_path, label)
        if os.path.exists(label_path):
            file_list = [
                os.path.join(label_path, filepath)
                for filepath in os.listdir(label_path)
            ]
            for image in file_list:
                # pred = model.predict(source=image)
                image = Image.open(image).convert("RGB")
                pred = model(np.array(image))
                pred_label = np.argmax([float(score) for score in list(pred[0].probs)])
                gt_class.append(int(class_id))
                pred_class.append(pred_label)
    return gt_class, pred_class


def log_confusion_to_experiment(experiment: Experiment, labelmap, matrix):
    confusion = {"categories": list(labelmap.values()), "values": matrix.tolist()}
    experiment.log(name="confusion", data=confusion, type="heatmap")
