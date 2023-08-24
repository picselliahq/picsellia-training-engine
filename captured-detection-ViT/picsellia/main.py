import os
import logging
from picsellia.sdk.experiment import Experiment

from datasets import load_dataset
from picsellia.types.enums import InferenceType
from transformers import AutoModelForObjectDetection, TrainingArguments, AutoImageProcessor
from transformers import Trainer
from transformers import TrainerCallback

from utils.picsellia import get_experiment, download_data, evaluate_asset, log_metrics
from utils.vit import CocoDetection, get_category_mapping, run_evaluation, get_filenames_by_ids, write_metadata_file, \
    read_annotation_file, get_category_mapping, format_coco_annot_to_jsonlines_format, transform_aug_ann, \
    custom_train_test_eval_split, collate_fn, save_annotation_file_images, format_evaluation_results, \
    get_dataset_image_ids, format_and_write_annotations

os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger('picsellia').setLevel(logging.INFO)

checkpoint = "facebook/detr-resnet-50"

experiment: Experiment = get_experiment()
dataset, data_dir = download_data(experiment=experiment)
annotations = format_and_write_annotations(dataset=dataset, data_dir=data_dir)

loaded_dataset = load_dataset("imagefolder", data_dir=data_dir)
train_test_valid_dataset = custom_train_test_eval_split(loaded_dataset=loaded_dataset, test_prop=0.2)

categories = get_category_mapping(annotations=annotations)
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}
labelmap = {str(i): category for i, category in enumerate(categories)}
experiment.log("labelmap", labelmap, "labelmap", replace=True)

image_processor = AutoImageProcessor.from_pretrained(checkpoint)
train_test_valid_dataset["train"] = train_test_valid_dataset["train"].with_transform(transform_aug_ann)
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
output_model_dir = os.path.join(experiment.checkpoint_dir)

training_args = TrainingArguments(
    output_dir=output_model_dir,
    per_device_train_batch_size=8,
    num_train_epochs=30,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    lr_scheduler_type='constant',
    learning_rate=1e-5,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)


class LogObjectDetectionMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            for metric_name, value in logs.items():
                log_metrics(metric_name=metric_name, value=value)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_test_valid_dataset["train"],
    tokenizer=image_processor,
    callbacks=[LogObjectDetectionMetricsCallback]
)
trainer.train()
trainer.save_model(output_dir=output_model_dir)

# test
image_processor = AutoImageProcessor.from_pretrained(output_model_dir)
model = AutoModelForObjectDetection.from_pretrained(output_model_dir)

path_output, path_anno = save_annotation_file_images(dataset=train_test_valid_dataset["test"], experiment=experiment,
                                                     id2label=id2label)
test_ds_coco_format = CocoDetection(path_output, image_processor, path_anno)

results = run_evaluation(test_ds_coco_format=test_ds_coco_format, im_processor=image_processor, model=model)
casted_results = format_evaluation_results(results=results)
experiment.log(name='evaluation metrics', type='table', data=casted_results)

# evaluate
dataset_labels = {label.name: label for label in dataset.list_labels()}
eval_image_ids = get_dataset_image_ids(train_test_valid_dataset, "eval")
id2filename_eval = get_filenames_by_ids(image_ids=eval_image_ids, annotations=annotations)

for file_path in list(id2filename_eval.values()):
    evaluate_asset(file_path=file_path)

experiment.compute_evaluations_metrics(inference_type=InferenceType.OBJECT_DETECTION)
