import logging
import os
import shutil

from picsellia import Client
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import InferenceType
from picsellia_tf2 import pxl_tf
from picsellia_tf2 import pxl_utils

from evaluator.tf_evaluator import (
    DetectionTensorflowEvaluator,
    SegmentationTensorflowEvaluator,
)

os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

logging.getLogger("picsellia").setLevel(logging.INFO)

if "api_token" not in os.environ:
    raise RuntimeError("You must set an api_token to run this image")

api_token = os.environ["api_token"]

if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]

if "organization_id" not in os.environ:
    organization_id = None
else:
    organization_id = os.environ["organization_id"]

client = Client(api_token=api_token, host=host, organization_id=organization_id)

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
    raise RuntimeError(
        "You must set the project_token or project_name and experiment_name"
    )

experiment.download_artifacts(with_tree=True)
parameters = experiment.get_log(name="parameters").data
attached_datasets = experiment.list_attached_dataset_versions()

if len(attached_datasets) == 3:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'train' dataset. Accepting 'train', 'test', 'val'"
        )
    try:
        test_ds = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'test' dataset. Accepting 'train', 'test', 'val'"
        )
    try:
        val_ds = experiment.get_dataset(name="val")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'val' dataset. Accepting 'train', 'test', 'val'"
        )

    labels = train_ds.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i + 1): label.name for i, label in enumerate(labels)}
    label_path = pxl_utils.generate_label_map(
        classes=label_names,
        output_path=experiment.base_dir,
    )

    for data_type, dataset in {
        "train": train_ds,
        "test": test_ds,
        "val": val_ds,
    }.items():
        dataset.download(
            target_path=os.path.join(experiment.png_dir, data_type), max_workers=8
        )
        stats = dataset.retrieve_stats()
        split = {
            "x": list(stats.label_repartition.keys()),
            "y": list(stats.label_repartition.values()),
        }

        annotation_path = dataset.build_coco_file_locally(
            enforced_ordered_categories=label_names
        )
        annotations = annotation_path.dict()
        categories_dict = [category["name"] for category in annotations["categories"]]
        for label in label_names:
            if label not in categories_dict:
                annotations["categories"].append(
                    {
                        "id": len(annotations["categories"]),
                        "name": label,
                        "supercategory": "",
                    }
                )

        if data_type == "train":
            train_split = split
            train_annotations, _, _ = pxl_utils.format_coco_file(
                imgdir=experiment.png_dir,
                annotations=annotations,
                train_assets=dataset.list_assets(),
            )
        elif data_type == "test":
            test_split = split
            _, _, test_annotations = pxl_utils.format_coco_file(
                imgdir=experiment.png_dir,
                annotations=annotations,
                train_assets=[],
                eval_assets=dataset.list_assets(),
                test_assets=[],
            )
        else:
            val_split = split
            _, val_annotations, _ = pxl_utils.format_coco_file(
                imgdir=experiment.png_dir,
                annotations=annotations,
                train_assets=[],
                eval_assets=[],
                test_assets=dataset.list_assets(),
            )
            val_assets = dataset.list_assets()

else:
    dataset = experiment.list_attached_dataset_versions()[0]
    prop = parameters.get("prop_train_split", 0.7)
    (
        train_assets,
        test_assets,
        val_assets,
        train_split,
        test_split,
        val_split,
        _,
    ) = dataset.train_test_val_split(
        ratios=[prop, (1.0 - prop) / 2, (1.0 - prop) / 2], random_seed=42
    )

    labels = dataset.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i + 1): label.name for i, label in enumerate(labels)}
    label_path = pxl_utils.generate_label_map(
        classes=label_names,
        output_path=experiment.base_dir,
    )

    annotation_path = dataset.build_coco_file_locally(
        enforced_ordered_categories=label_names
    )
    annotations = annotation_path.dict()
    categories_dict = [category["name"] for category in annotations["categories"]]
    for label in label_names:
        if label not in categories_dict:
            annotations["categories"].append(
                {
                    "id": len(annotations["categories"]),
                    "name": label,
                    "supercategory": "",
                }
            )

    for data_type, assets in {
        "train": train_assets,
        "test": test_assets,
        "val": val_assets,
    }.items():
        assets.download(
            target_path=os.path.join(experiment.png_dir, data_type), max_workers=8
        )

    test_ds = dataset

    train_annotations, test_annotations, val_annotations = pxl_utils.format_coco_file(
        imgdir=experiment.png_dir,
        annotations=annotations,
        train_assets=train_assets,
        eval_assets=test_assets,
        test_assets=val_assets,
    )

experiment.log("labelmap", labelmap, "labelmap", replace=True)
experiment.log(
    "train-split", pxl_utils.sort_split(train_split, label_names), "bar", replace=True
)
experiment.log(
    "test-split", pxl_utils.sort_split(test_split, label_names), "bar", replace=True
)
experiment.log(
    "val-split", pxl_utils.sort_split(val_split, label_names), "bar", replace=True
)

print("\n")
experiment.start_logging_chapter("Create records")

pxl_utils.create_record_files(
    train_annotations=train_annotations,
    eval_annotations=test_annotations,
    test_annotations=val_annotations,
    label_path=label_path,
    record_dir=experiment.record_dir,
    tfExample_generator=pxl_tf.tf_vars_generator,
    annotation_type=parameters["annotation_type"],
)

# edit training config
training_config_dir = experiment.config_dir
test_config = os.path.join(experiment.base_dir, "test_config")
if not os.path.exists(test_config):
    os.makedirs(test_config)
    
if os.path.isfile(os.path.join(training_config_dir, "pipeline.config")):
    shutil.copy(
        os.path.join(training_config_dir, "pipeline.config"),
        os.path.join(test_config, "pipeline.config"),
    )

pxl_utils.edit_config(
    model_selected=experiment.checkpoint_dir,
    input_config_dir=training_config_dir,
    output_config_dir=training_config_dir,
    train_record_path=os.path.join(experiment.record_dir, "train.record"),
    eval_record_path=os.path.join(experiment.record_dir, "val.record"),
    label_map_path=label_path,
    num_steps=parameters["steps"],
    batch_size=parameters["batch_size"],
    learning_rate=parameters["learning_rate"],
    annotation_type=parameters["annotation_type"],
    parameters=parameters,
)

# edit final test config

pxl_utils.edit_config(
    model_selected=experiment.checkpoint_dir,
    input_config_dir=test_config,
    output_config_dir=test_config,
    train_record_path=os.path.join(experiment.record_dir, "train.record"),
    eval_record_path=os.path.join(experiment.record_dir, "test.record"),
    label_map_path=label_path,
    num_steps=parameters["steps"],
    batch_size=parameters["batch_size"],
    learning_rate=parameters["learning_rate"],
    annotation_type=parameters["annotation_type"],
    parameters=parameters,
)

print("\n")
experiment.start_logging_chapter("Start training")

pxl_utils.train(
    ckpt_dir=experiment.checkpoint_dir,
    config_dir=experiment.config_dir,
    log_real_time=experiment,
    evaluate_fn=pxl_utils.evaluate,
    log_metrics=pxl_utils.log_metrics,
    checkpoint_every_n=parameters.get("checkpoint_every_n", 10),
)

print("\n")
experiment.start_logging_chapter("Store artifacts")

pxl_utils.export_graph(
    ckpt_dir=experiment.checkpoint_dir,
    exported_model_dir=experiment.exported_model_dir,
    config_dir=training_config_dir,
)
experiment.store("model-latest")
experiment.store("config")
experiment.store("checkpoint-data-latest")
experiment.store("checkpoint-index-latest")

print("\n")
experiment.start_logging_chapter("Computing metrics on test dataset")

experiment.start_logging_buffer(9)

test_metrics_dir = os.path.join(experiment.base_dir, "test_metrics")
if not os.path.exists(test_metrics_dir):
    os.makedirs(test_metrics_dir)

pxl_utils.evaluate(test_metrics_dir, test_config, experiment.checkpoint_dir)

metrics = pxl_utils.tf_events_to_dict("{}/test_metrics".format(experiment.name), "test")
experiment.log("Evaluation/Metrics", metrics, "table", replace=True)

conf, _ = pxl_utils.get_confusion_matrix(
    input_tfrecord_path=os.path.join(experiment.record_dir, "test.record"),
    model=os.path.join(experiment.exported_model_dir, "saved_model"),
    labelmap=labelmap,
)

confusion = {"categories": list(labelmap.values()), "values": conf.tolist()}
experiment.log("Evaluation/confusion-matrix", confusion, "heatmap", replace=True)
experiment.end_logging_buffer()

print("\n")
experiment.start_logging_chapter("Starting Evaluation")

inference_type = experiment.get_base_model_version().type

if inference_type == InferenceType.OBJECT_DETECTION:
    detection_evaluator = DetectionTensorflowEvaluator(
        experiment=experiment,
        dataset=test_ds,
        asset_list=test_assets,
        confidence_threshold=0.1,
    )

    detection_evaluator.evaluate()

elif inference_type == InferenceType.SEGMENTATION:
    segmentation_evaluator = SegmentationTensorflowEvaluator(
        experiment=experiment,
        dataset=test_ds,
        asset_list=test_assets,
        confidence_threshold=0.1,
    )
    segmentation_evaluator.evaluate()


else:
    print(
        "The only supported inference types for evaluation are object detection and segmentation. "
        "Please add inference type to model if you haven't already"
    )
