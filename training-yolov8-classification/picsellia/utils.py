from typing import Tuple, Any

import numpy
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
    _create_class_directories(coco=coco, base_imdir=base_imdir)
    for i in coco.imgs:
        image = coco.imgs[i]
        cat = get_image_annotation(coco=coco, fnames=fnames, image=image)
        if cat is None:
            continue
        move_image(
            filename=image["file_name"],
            old_location_path=base_imdir,
            new_location_path=os.path.join(base_imdir, cat["name"]),
        )
    print(f"Formatting {base_imdir} .. OK")
    return base_imdir


def _create_class_directories(coco: COCO, base_imdir: str = None):
    for i in coco.cats:
        cat = coco.cats[i]
        class_folder = os.path.join(base_imdir, cat["name"])
        if not os.path.isdir(class_folder):
            os.makedirs(class_folder)
    print(f"Formatting {base_imdir} ..")


def get_image_annotation(coco: COCO, fnames: list[str], image: dict) -> None | dict:
    if image["file_name"] not in fnames:
        return None
    ann = coco.loadAnns(coco.getAnnIds(image["id"]))
    if len(ann) > 1:
        print(f"{image['file_name']} has more than one class. Skipping")
    ann = ann[0]
    cat = coco.loadCats(ann["category_id"])[0]

    return cat


def move_image(filename: str, old_location_path: str, new_location_path: str):
    old_path = os.path.join(old_location_path, filename)
    new_path = os.path.join(new_location_path, filename)
    try:
        shutil.move(old_path, new_path)
    except Exception as e:
        print(f"{filename} skipped.")


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


def get_prop_parameter(parameters: dict):
    prop = parameters.get("prop_train_split", 0.7)
    return prop


def make_train_test_val_dirs():
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)


def move_images_in_train_test_val_folders(train_assets, test_assets, eval_assets):
    for asset in train_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/train",
        )
    for asset in test_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/test",
        )

    for asset in eval_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/val",
        )


def move_image(filename: str, old_location_path: str, new_location_path: str):
    old_path = os.path.join(old_location_path, filename)
    new_path = os.path.join(new_location_path, filename)
    shutil.move(old_path, new_path)


def split_single_dataset(experiment: Experiment, train_set: DatasetVersion):
    parameters = experiment.get_log("parameters").data
    prop = get_prop_parameter(parameters)
    (
        train_assets,
        test_assets,
        eval_assets,
        train_rep,
        test_rep,
        val_rep,
        labels,
    ) = train_set.train_test_val_split([prop, (1 - prop) / 2, (1 - prop) / 2])

    make_train_test_val_dirs()
    move_images_in_train_test_val_folders(
        train_assets=train_assets, test_assets=test_assets, eval_assets=eval_assets
    )

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


def format_confusion_matrix(labelmap: dict, matrix: numpy.ndarray) -> dict:
    return {"categories": list(labelmap.values()), "values": matrix.tolist()}


def log_confusion_to_experiment(
    experiment: Experiment, labelmap: dict, matrix: numpy.ndarray
):
    confusion = format_confusion_matrix(labelmap=labelmap, matrix=matrix)
    experiment.log(name="confusion", data=confusion, type="heatmap")
