from pycocotools.coco import COCO
import os
import numpy as np
import evaluate
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.types.enums import AnnotationFileType
from picsellia import Experiment

import shutil


def get_predicted_label_confidence(predictions):
    scores = []
    classes = []
    for pred in predictions:
        scores.append(pred['score'])
        classes.append(pred['label'])

    max_conf = max(scores)

    predicted_class = classes[scores.index(max_conf)]

    return predicted_class, max_conf


def compute_metrics(eval_pred):

    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)


def prepare_datasets_with_annotation(experiment: Experiment, train_set: DatasetVersion, test_set: DatasetVersion, val_set: DatasetVersion):
    coco_train, coco_test, coco_val = _create_coco_objects(
        train_set, test_set, val_set)

    move_files_in_class_directories(
        coco_train, os.path.join(experiment.base_dir, "data/train"))
    move_files_in_class_directories(
        coco_test, os.path.join(experiment.base_dir, "data/test"))
    move_files_in_class_directories(
        coco_val, os.path.join(experiment.base_dir, "data/val"))

    evaluation_ds = val_set
    evaluation_assets = evaluation_ds.list_assets()

    return evaluation_ds, evaluation_assets


def _create_coco_objects(train_set: DatasetVersion, test_set: DatasetVersion, val_set: DatasetVersion):
    train_annotation_path = train_set.export_annotation_file(
        AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)

    test_annotation_path = test_set.export_annotation_file(
        AnnotationFileType.COCO)
    coco_test = COCO(test_annotation_path)

    val_annotation_path = val_set.export_annotation_file(
        AnnotationFileType.COCO)
    coco_val = COCO(val_annotation_path)

    return coco_train, coco_test, coco_val


def move_files_in_class_directories(coco: COCO, base_imdir: str = None) -> None:
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
        cat = coco.loadCats(ann['category_id'])[0]
        fpath = os.path.join(base_imdir, im['file_name'])
        new_fpath = os.path.join(base_imdir, cat['name'], im['file_name'])
        try:
            shutil.move(fpath, new_fpath)
            pass
        except Exception as e:
            print(f"{im['file_name']} skipped.")
    print(f"Formatting {base_imdir} .. OK")
    return base_imdir
