import os
import picsellia

from picsellia.sdk.experiment import Experiment
from picsellia.sdk.dataset import DatasetVersion
from utils.picsellia import download_data, evaluate_asset
from picsellia.types.enums import InferenceType
from datasets import load_dataset, DatasetDict

import transformers
from transformers import (
    AutoModelForObjectDetection,
    TrainingArguments,
    AutoImageProcessor,
    Trainer,
)

from utils.vit import (
    CocoDetection,
    run_evaluation,
    get_filenames_by_ids,
    transform_images_and_annotations,
    custom_train_test_eval_split,
    get_id2label_mapping,
    collate_fn,
    save_annotation_file_images,
    format_evaluation_results,
    get_dataset_image_ids,
    format_and_write_annotations,
    log_labelmap,
)


class TrainingPipeline:
    checkpoint = "facebook/detr-resnet-50"

    def __init__(self):
        self.experiment = self.get_experiment()
        self.annotations = {}
        self.output_model_dir = os.path.join(self.experiment.checkpoint_dir)
        self.data_dir = os.path.join(self.experiment.base_dir, "data")
        self.id2label = {}

    def get_experiment(self) -> Experiment:
        if "api_token" not in os.environ:
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

        client = picsellia.Client(
            api_token=api_token, host=host, organization_id=organization_id
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
            Exception(
                "You must set the project_token or project_name and experiment_name"
            )
        return experiment

    def prepare_data_for_training(
        self,
    ) -> tuple[DatasetDict, DatasetVersion]:
        dataset = download_data(experiment=self.experiment)
        self.annotations = format_and_write_annotations(
            dataset=dataset, data_dir=self.data_dir
        )
        loaded_dataset = load_dataset("imagefolder", data_dir=self.data_dir)
        train_test_valid_dataset = custom_train_test_eval_split(
            loaded_dataset=loaded_dataset, test_prop=0.2
        )
        train_test_valid_dataset["train"] = train_test_valid_dataset[
            "train"
        ].with_transform(transform_images_and_annotations)
        self.id2label = get_id2label_mapping(annotations=self.annotations)
        log_labelmap(id2label=self.id2label)
        return train_test_valid_dataset, dataset

    def train(
        self, train_test_valid_dataset: DatasetDict, callback_list: list
    ) -> transformers.trainer.Trainer:
        label2id = {v: k for k, v in self.id2label.items()}

        image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        model = AutoModelForObjectDetection.from_pretrained(
            self.checkpoint,
            id2label=self.id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        training_args = TrainingArguments(
            output_dir=self.output_model_dir,
            per_device_train_batch_size=8,
            num_train_epochs=30,
            fp16=True,
            save_steps=200,
            logging_steps=50,
            lr_scheduler_type="constant",
            learning_rate=1e-5,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=train_test_valid_dataset["train"],
            tokenizer=image_processor,
            callbacks=callback_list,
        )
        trainer.train()

        return trainer

    def test(
        self,
        train_test_valid_dataset: DatasetDict,
    ) -> tuple[transformers.models, transformers.models]:
        image_processor = AutoImageProcessor.from_pretrained(self.output_model_dir)
        model = AutoModelForObjectDetection.from_pretrained(self.output_model_dir)

        path_output, path_anno = save_annotation_file_images(
            dataset=train_test_valid_dataset["test"],
            experiment=self.experiment,
            id2label=self.id2label,
        )
        test_ds_coco_format = CocoDetection(path_output, image_processor, path_anno)

        results = run_evaluation(
            test_ds_coco_format=test_ds_coco_format,
            im_processor=image_processor,
            model=model,
        )
        casted_results = format_evaluation_results(results=results)
        self.experiment.log(
            name="evaluation metrics", type="table", data=casted_results
        )
        return image_processor, model

    def evaluate(
        self,
        dataset: DatasetVersion,
        train_test_valid_dataset: DatasetDict,
        model: transformers.models,
    ):
        dataset_labels = {label.name: label for label in dataset.list_labels()}
        eval_image_ids = get_dataset_image_ids(train_test_valid_dataset, "eval")
        id2filename_eval = get_filenames_by_ids(
            image_ids=eval_image_ids, annotations=self.annotations
        )

        for file_path in list(id2filename_eval.values()):
            evaluate_asset(
                file_path=file_path,
                data_dir=self.data_dir,
                experiment=self.experiment,
                dataset_labels=dataset_labels,
                model=model,
            )

        self.experiment.compute_evaluations_metrics(
            inference_type=InferenceType.OBJECT_DETECTION
        )
