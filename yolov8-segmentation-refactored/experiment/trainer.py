import os

from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.dataset import DatasetVersion
from pycocotools.coco import COCO

import segmentation_trainer
from abstract_trainer.trainer import AbstractTrainer
from evaluator.yolo_evaluator import SegmentationYOLOEvaluator
from utils import (
    get_train_test_eval_datasets_from_experiment,
    write_annotation_file,
    create_yolo_segmentation_label,
    get_prop_parameter,
    log_split_dataset_repartition_to_experiment,
    generate_data_yaml,
    setup_hyp,
    send_run_to_picsellia,
)


class Yolov8SegmentationTrainer(AbstractTrainer):
    def __init__(self):
        super().__init__()
        self.evaluation_ds = None
        self.evaluation_assets = None
        self.annotations_dict = None
        self.annotations_coco = None
        self.label_names = None
        self.current_dir = os.path.join(os.getcwd(), self.experiment.base_dir)
        self.annotations_path = "annotations.json"

    def prepare_data_for_training(self):
        self.experiment.download_artifacts(with_tree=True)
        (
            has_three_datasets,
            train_set,
            test_set,
            eval_set,
        ) = get_train_test_eval_datasets_from_experiment(self.experiment)

        if train_set is not None:
            labels = train_set.list_labels()
            self.label_names = [label.name for label in labels]
            self.labelmap = {str(i): label.name for i, label in enumerate(labels)}
            self.experiment.log("labelmap", self.labelmap, "labelmap", replace=True)
        if has_three_datasets:
            for data_type, dataset in {
                "train": train_set,
                "test": test_set,
                "val": eval_set,
            }.items():
                self._prepare_annotations_label(data_type=data_type, dataset=dataset)
            self.evaluation_ds = test_set
            self.evaluation_assets = self.evaluation_ds.list_assets()
        else:
            self._process_single_dataset(train_set=train_set)

    def _process_single_dataset(self, train_set: DatasetVersion):
        self.annotations_dict = self._make_annotation_dict_by_dataset(dataset=train_set)
        write_annotation_file(
            annotations_dict=self.annotations_dict,
            annotations_path=self.annotations_path,
        )
        self.annotations_coco = COCO(self.annotations_path)
        prop = get_prop_parameter(parameters=self.parameters)
        (
            train_assets,
            test_assets,
            eval_assets,
            train_rep,
            test_rep,
            val_rep,
            labels,
        ) = train_set.train_test_val_split(([prop, (1 - prop) / 2, (1 - prop) / 2]))

        for data_type, assets in {
            "train": train_assets,
            "val": eval_assets,
            "test": test_assets,
        }.items():
            self._download_data_with_label(assets=assets, data_type=data_type)
        self.evaluation_ds = train_set
        self.evaluation_assets = test_assets
        log_split_dataset_repartition_to_experiment(
            experiment=self.experiment,
            labelmap=self.labelmap,
            train_rep=train_rep,
            test_rep=test_rep,
            val_rep=val_rep,
        )

    def _prepare_annotations_label(self, data_type: str, dataset: DatasetVersion):
        self.annotations_dict = self._make_annotation_dict_by_dataset(dataset=dataset)
        write_annotation_file(
            annotations_dict=self.annotations_dict,
            annotations_path=self.annotations_path,
        )
        self.annotations_coco = COCO(self.annotations_path)
        self._download_data_with_label(
            assets=dataset.list_assets(), data_type=data_type
        )

    def _make_annotation_dict_by_dataset(self, dataset: DatasetVersion) -> dict:
        coco_annotation = dataset.build_coco_file_locally(
            enforced_ordered_categories=self.label_names
        )
        annotations_dict = coco_annotation.dict()
        categories_dict = [
            category["name"] for category in annotations_dict["categories"]
        ]
        for label in self.label_names:
            if label not in categories_dict:
                annotations_dict["categories"].append(
                    {
                        "id": len(annotations_dict["categories"]),
                        "name": label,
                        "supercategory": "",
                    }
                )
        return annotations_dict

    def _download_data_with_label(self, assets: MultiAsset, data_type: str):
        assets.download(
            target_path=os.path.join(self.experiment.png_dir, data_type, "images"),
            max_workers=8,
        )
        create_yolo_segmentation_label(
            self.experiment,
            data_type,
            self.annotations_dict,
            self.annotations_coco,
            self.label_names,
        )

    def train(self):
        data_yaml_path = generate_data_yaml(
            experiment=self.experiment,
            labelmap=self.labelmap,
            config_path=self.current_dir,
        )
        config = setup_hyp(
            experiment=self.experiment,
            data_yaml_path=data_yaml_path,
            params=self.parameters,
            label_map=self.labelmap,
            cwd=self.current_dir,
            task="segment",
        )

        trainer = segmentation_trainer.PicselliaSegmentationTrainer(
            experiment=self.experiment, cfg=config, parameters=self.parameters
        )
        trainer.train()
        send_run_to_picsellia(self.experiment, self.current_dir, trainer.save_dir)

    def eval(self):
        segmentation_evaluator = SegmentationYOLOEvaluator(
            experiment=self.experiment,
            dataset=self.evaluation_ds,
            asset_list=self.evaluation_assets,
            confidence_threshold=self.parameters.get("confidence_threshold", 0.1),
        )

        segmentation_evaluator.evaluate()
