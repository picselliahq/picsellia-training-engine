import logging
import os
import shutil

import evaluate
from datasets import load_dataset
from picsellia import Client
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import AnnotationFileType, InferenceType
from pycocotools.coco import COCO
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import (
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers import pipeline, AutoImageProcessor, DefaultDataCollator

from utils import (
    move_files_in_class_directories,
    get_predicted_label_confidence,
    compute_metrics,
    prepare_datasets_with_annotation
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

client = Client(api_token=api_token, host=host,
                organization_id=organization_id)

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

parameters = experiment.get_log("parameters").data
dataset_list = experiment.list_attached_dataset_versions()


if len(dataset_list) == 1:
    dataset = dataset_list[0]
    dataset.download("images")

    train_annotation_path = dataset.export_annotation_file(
        AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)
    # prop = parameters.get("prop", 0.7)
    prop = 0.7

    (
        train_assets,
        test_assets,
        eval_assets,
        train_rep,
        test_rep,
        val_rep,
        labels,
    ) = dataset.train_test_val_split([prop, (1 - prop) / 2, (1 - prop) / 2])

    train_path = os.path.join(experiment.base_dir, "data/train")
    test_path = os.path.join(experiment.base_dir, "data/test")
    eval_path = os.path.join(experiment.base_dir, "data/val")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    for asset in train_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join(train_path, asset.filename)
        shutil.move(old_path, new_path)

    for asset in test_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join(test_path, asset.filename)
        shutil.move(old_path, new_path)

    for asset in eval_assets:
        old_path = os.path.join("images", asset.filename)
        new_path = os.path.join(eval_path, asset.filename)
        shutil.move(old_path, new_path)
    move_files_in_class_directories(coco_train, train_path)
    move_files_in_class_directories(coco_train, test_path)
    move_files_in_class_directories(coco_train, eval_path)

elif len(dataset_list) == 3:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                            accepting 'train', 'test', 'eval'")
    try:
        test_ds = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                            accepting 'train', 'test', 'eval'")
    try:
        eval_ds = experiment.get_dataset(name="eval")
    except Exception:
        raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                accepting 'train', 'test', 'eval'")

    for data_type, dataset in {'train': train_ds, 'test': test_ds}.items():
        dataset.download(
            target_path=os.path.join(experiment.base_dir, "data", data_type), max_workers=8
        )
    eval_ds.download(target_path=os.path.join(
        experiment.base_dir, "data/val"), max_workers=8)

    evaluation_ds, evaluation_assets = prepare_datasets_with_annotation(
        experiment, train_ds, test_ds, eval_ds)

    eval_path = os.path.join(experiment.base_dir, "data/val")
    dataset = evaluation_ds


loaded_dataset = load_dataset(
    "imagefolder", data_dir=os.path.join(experiment.base_dir, "data")
)
dataset_labels = {label.name: label for label in dataset.list_labels()}

labels = loaded_dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

experiment.log("labelmap", id2label, "labelmap", replace=True)

# download model's artifacts if available
try:
    checkpoint_file = experiment.get_artifact("pytorch_model")
    loaded_checkpoint_folder_path = experiment.base_dir
    experiment.download_artifacts(with_tree=False)
except ResourceNotFoundError as e:
    print(e)
    loaded_checkpoint_folder_path = None

# load ViT preprocessor to process images into a Tensor
checkpoint = "google/vit-base-patch16-224-in21k"
if loaded_checkpoint_folder_path:
    checkpoint = loaded_checkpoint_folder_path
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


def transforms(examples):
    normalize = Normalize(mean=image_processor.image_mean,
                          std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    examples["pixel_values"] = [_transforms(
        img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


loaded_dataset = loaded_dataset.with_transform(transforms)

data_collator = DefaultDataCollator()
accuracy = evaluate.load("accuracy")

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

output_model_dir = os.path.join(experiment.checkpoint_dir)


class LogMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):

        if state.is_local_process_zero:
            for metric_name, value in logs.items():
                if metric_name in ['train_loss', 'total_flos', 'train_steps_per_second', 'train_samples_per_second', 'train_runtime']:
                    experiment.log(str(metric_name), float(value), 'value')
                else:
                    experiment.log(str(metric_name), float(value), 'line')


nbr_epochs = int(parameters.get("epochs", 5))
batch_size = int(parameters.get("batch_size", 16))
learning_rate = parameters.get("learning_rate", 5e-5)

training_args = TrainingArguments(
    output_dir=output_model_dir,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=nbr_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    save_total_limit=2,
    save_strategy="no",
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=loaded_dataset["train"],
    eval_dataset=loaded_dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    callbacks=[LogMetricsCallback],
)

trainer.train()

trainer.save_model(output_dir=output_model_dir)

for artifact in os.listdir(path=output_model_dir):
    experiment.store(
        name=artifact.split(".")[0], path=os.path.join(
            output_model_dir, artifact)
    )


# evaluate
classifier = pipeline("image-classification", model=output_model_dir)

for path, subdirs, file_list in os.walk(eval_path):
    if file_list != []:
        for file in file_list:
            file_path = os.path.join(path, file)
            current_prediction = classifier(str(file_path))

            pred_label, pred_conf = get_predicted_label_confidence(
                current_prediction)

            asset_filename = file_path.split("/")[-1]

            try:
                asset = dataset.find_asset(filename=asset_filename)
            except Exception as e:
                print(e)

            experiment.add_evaluation(
                asset=asset,
                classifications=[
                    (dataset_labels[pred_label], float(pred_conf))],
            )
            print(f"Asset: {asset_filename} evaluated.")

experiment.compute_evaluations_metrics(InferenceType.CLASSIFICATION)
