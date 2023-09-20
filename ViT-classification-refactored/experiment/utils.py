import os
import shutil
import logging

import evaluate
import numpy as np
from picsellia import Experiment
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.types.enums import AnnotationFileType
from pycocotools.coco import COCO
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.label import Label


def get_predicted_label_confidence(predictions):
    scores = []
    classes = []
    for pred in predictions:
        scores.append(pred["score"])
        classes.append(pred["label"])

    max_conf = max(scores)

    predicted_class = classes[scores.index(max_conf)]

    return predicted_class, max_conf


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)


def prepare_datasets_with_annotation(
    train_set: DatasetVersion,
    test_set: DatasetVersion,
    val_set: DatasetVersion,
    train_test_eval_path_dict: dict,
):
    coco_train, coco_test, coco_val = _create_coco_objects(train_set, test_set, val_set)

    move_files_in_class_directories(coco_train, train_test_eval_path_dict["train_path"])
    move_files_in_class_directories(coco_test, train_test_eval_path_dict["test_path"])
    move_files_in_class_directories(coco_val, train_test_eval_path_dict["eval_path"])

    evaluation_ds = val_set
    evaluation_assets = evaluation_ds.list_assets()

    return evaluation_ds, evaluation_assets


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


def _move_all_files_in_class_directories(
    train_set: DatasetVersion, train_test_eval_path_dict: dict
) -> None:
    train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)
    _move_files_in_class_directories(
        coco_train, train_test_eval_path_dict["train_path"]
    )
    _move_files_in_class_directories(coco_train, train_test_eval_path_dict["test_path"])
    _move_files_in_class_directories(coco_train, train_test_eval_path_dict["eval_path"])


def _move_files_in_class_directories(coco: COCO, base_imdir: str = None) -> None | str:
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
    logging.info(f"Formatting {base_imdir} .. OK")
    return base_imdir


def _create_class_directories(coco: COCO, base_imdir: str = None) -> None:
    for i in coco.cats:
        cat = coco.cats[i]
        class_folder = os.path.join(base_imdir, cat["name"])
        if not os.path.isdir(class_folder):
            os.makedirs(class_folder)
    logging.info(f"Formatting {base_imdir} ..")


def get_image_annotation(coco: COCO, fnames: list[str], image: dict) -> None | dict:
    if image["file_name"] not in fnames:
        return None
    ann = coco.loadAnns(coco.getAnnIds(image["id"]))
    if len(ann) > 1:
        logging.info(f"{image['file_name']} has more than one class. Skipping")
    ann = ann[0]
    cat = coco.loadCats(ann["category_id"])[0]

    return cat


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


def download_triple_dataset(
    train_set: DatasetVersion, test_set: DatasetVersion, eval_set: DatasetVersion
) -> None:
    for data_type, dataset in {
        "train": train_set,
        "test": test_set,
        "val": eval_set,
    }.items():
        dataset.download(target_path=os.path.join("data", data_type), max_workers=8)


def get_train_test_eval_datasets_from_experiment(
    experiment: Experiment,
) -> tuple[bool, bool, DatasetVersion, DatasetVersion, DatasetVersion]:
    number_of_attached_datasets = len(experiment.list_attached_dataset_versions())
    has_three_datasets, has_two_datasets = False, False
    if number_of_attached_datasets == 3:
        has_three_datasets = True
        train_set, test_set, eval_set = _get_three_attached_datasets(experiment)
    elif number_of_attached_datasets == 2:
        has_two_datasets = True
        train_set, test_set, eval_set = _transform_two_attached_datasets_to_three(
            experiment
        )
    elif number_of_attached_datasets == 1:
        logging.info(
            "We only found one dataset inside your experiment, the train/test/split will be performed automatically."
        )
        train_set: DatasetVersion = experiment.list_attached_dataset_versions()[0]
        test_set = None
        eval_set = None

    else:
        logging.info(
            "We need at least 1 and at most 3 datasets attached to this experiment "
        )

    return has_two_datasets, has_three_datasets, train_set, test_set, eval_set


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


def split_single_dataset(
    parameters: dict, train_set: DatasetVersion, train_test_eval_path_dict: dict
) -> tuple[
    MultiAsset,
    MultiAsset,
    MultiAsset,
    dict[str, list],
    dict[str, list],
    dict[str, list],
    list[Label],
]:
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

    make_train_test_val_dirs(train_test_eval_path_dict)
    move_images_in_train_test_val_folders(
        train_assets=train_assets,
        test_assets=test_assets,
        eval_assets=eval_assets,
        train_test_val_path=train_test_eval_path_dict,
    )

    return train_assets, test_assets, eval_assets, train_rep, test_rep, val_rep, labels


def get_prop_parameter(parameters: dict) -> float:
    prop = parameters.get("prop_train_split", 0.7)
    return prop


def make_train_test_val_dirs(train_test_eval_path_dict: dict) -> None:
    os.makedirs(train_test_eval_path_dict["train_path"], exist_ok=True)
    os.makedirs(train_test_eval_path_dict["test_path"], exist_ok=True)
    os.makedirs(train_test_eval_path_dict["eval_path"], exist_ok=True)


def move_images_in_train_test_val_folders(
    train_assets: MultiAsset,
    test_assets: MultiAsset,
    eval_assets: MultiAsset,
    train_test_val_path: dict,
) -> None:
    for asset in train_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path=train_test_val_path["train_path"],
        )
    for asset in test_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path=train_test_val_path["test_path"],
        )

    for asset in eval_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path=train_test_val_path["eval_path"],
        )


def move_image(filename: str, old_location_path: str, new_location_path: str) -> None:
    old_path = os.path.join(old_location_path, filename)
    new_path = os.path.join(new_location_path, filename)
    try:
        shutil.move(old_path, new_path)
    except Exception as e:
        logging.info(f"{filename} skipped.")


def transforms(examples):
    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    examples["pixel_values"] = [
        _transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples
