import os
import numpy as np
import shutil
import unittest
from utils import (
    get_prop_parameter,
    make_train_test_val_dirs,
    move_image,
    _get_three_attached_datasets,
    _create_coco_objects,
    _create_class_directories,
    get_image_annotation,
    _transform_two_attached_datasets_to_three,
    get_train_test_eval_datasets_from_experiment,
    format_confusion_matrix,
    order_repartition_according_labelmap,
)
from helpers import Yolov8ClassificationTrainer
from picsellia import Client
from datetime import date
import time

TOKEN = os.environ["TEST_TOKEN"]
ORGA_NAME = os.environ["TEST_ORGA"]


class TestYoloClassificationUtils(unittest.TestCase):
    eval_set = None
    test_set = None
    train_set = None
    dataset = None
    experiment = None
    project = None
    client = None
    organization_name: str
    token: str
    labelmap = {"0": "covid", "1": "normal", "2": "pneumonia"}

    @classmethod
    def setUpClass(cls) -> None:
        cls.token = TOKEN
        cls.organization_name = ORGA_NAME
        cls.client = Client(
            api_token=cls.token, organization_name=cls.organization_name
        )
        cls.project = cls.client.create_project(
            name=f"test_yolo_classif-utils{str(date.today())}-{str(time.time())}"
        )
        cls.model_version = cls.client.get_public_model(
            "yolov8-classification"
        ).get_version(0)
        cls.experiment = cls.project.create_experiment(name="yolo-triple-dataset")
        cls.dataset = cls.client.get_dataset_by_id(
            "01888c1b-cfb6-768b-83dd-2e1c460e79cf"
        )
        cls.experiment.attach_dataset(
            name="train", dataset_version=cls.dataset.get_version("train")
        )
        cls.experiment.attach_dataset(
            name="test", dataset_version=cls.dataset.get_version("test")
        )
        cls.experiment.attach_dataset(
            name="eval", dataset_version=cls.dataset.get_version("val")
        )
        cls.train_set, cls.test_set, cls.eval_set = _get_three_attached_datasets(
            cls.experiment
        )
        cls.coco_train, cls.coco_test, cls.coco_val = _create_coco_objects(
            cls.train_set, cls.test_set, cls.eval_set
        )
        cls.train_path = "data/train"
        cls.test_path = "data/test"
        cls.val_path = "data/val"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()
        sample_file_path = "data/test/grape_image.jpeg"
        if os.path.isfile(sample_file_path):
            shutil.move(sample_file_path, "test_files")

    def test_get_prop_parameters(self):
        parameters = {"epochs": 100, "patience": 500, "image_size": 640}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.7)

    def test_get_prop_parameters_exit(self):
        parameters = {"epochs": 100, "patience": 500, "prop_train_split": 0.8}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.8)

    def test_make_train_test_val_dirs(self):
        make_train_test_val_dirs()
        is_created = False
        if (
            os.path.exists(self.train_path)
            and os.path.exists(self.test_path)
            and os.path.exists(self.val_path)
        ):
            is_created = True
        self.assertTrue(is_created)

        move_image(
            filename="grape_image.jpeg",
            old_location_path="test_files",
            new_location_path=self.test_path,
        )
        is_moved = False
        if os.path.isfile(os.path.join(self.test_path, "grape_image.jpeg")):
            is_moved = True
        self.assertTrue(is_moved)

    def test_create_class_directories(self):
        _create_class_directories(coco=self.coco_train, base_imdir=self.train_path)
        is_created = False
        if (
            os.path.exists(os.path.join(self.train_path, "covid"))
            and os.path.exists(os.path.join(self.train_path, "normal"))
            and os.path.exists(os.path.join(self.train_path, "pneumonia"))
        ):
            is_created = True
        self.assertTrue(is_created)

    def test_get_image_annotation(self):
        fnames = ["covid_082.jpg"]
        image = {
            "id": 0,
            "file_name": "covid_082.jpg",
            "width": 1165,
            "height": 1163,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None,
        }
        cat = get_image_annotation(coco=self.coco_train, fnames=fnames, image=image)
        expected_cat = {"id": 0, "name": "covid", "supercategory": None}

        self.assertEqual(expected_cat, cat)

    def test_get_three_attached_datasets(self):
        train_set, test_set, eval_test = _get_three_attached_datasets(self.experiment)
        self.assertEqual(
            (self.train_set, self.test_set, self.eval_set),
            (train_set, test_set, eval_test),
        )

    def test_transform_two_attached_datasets_to_three(self):
        train_set, test_set, eval_set = _transform_two_attached_datasets_to_three(
            experiment=self.experiment
        )
        self.assertEqual(
            (self.train_set, self.test_set, self.test_set),
            (train_set, test_set, eval_set),
        )

    def test_get_train_eval_datasets_from_experiment(self):
        results = get_train_test_eval_datasets_from_experiment(self.experiment)
        expected_results = (False, True, self.train_set, self.test_set, self.eval_set)
        self.assertEqual(expected_results, results)

    def test_format_confusion_matrix(self):
        matrix = np.array([[60, 0, 0], [46, 0, 0], [39, 0, 0]])
        formatted_matrix = format_confusion_matrix(
            labelmap=self.labelmap, matrix=matrix
        )
        expected_matrix = {
            "categories": ["covid", "normal", "pneumonia"],
            "values": [[60, 0, 0], [46, 0, 0], [39, 0, 0]],
        }
        self.assertEqual(expected_matrix, formatted_matrix)

    def test_order_repartition_according_labelmap(self):
        repartition = {"x": ["covid", "pneumonia", "normal"], "y": [12, 11, 7]}
        ordered_repartition = order_repartition_according_labelmap(
            labelmap=self.labelmap,
            repartition=repartition,
        )
        expected_results = {"x": ["covid", "normal", "pneumonia"], "y": [12, 7, 11]}

        self.assertEqual(expected_results, ordered_repartition)


class TestYoloClassificationHelpers(unittest.TestCase):
    model_version = None
    dataset = None
    organization_name: str
    token: str
    experiment = None
    client = None
    project = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.token = TOKEN
        cls.organization_name = ORGA_NAME
        cls.client = Client(
            api_token=cls.token, organization_name=cls.organization_name
        )
        cls.project = cls.client.create_project(
            name=f"test_yolo_classif-helpers{str(date.today())}-{str(time.time())}"
        )
        cls.model_version = cls.client.get_public_model(
            "yolov8-classification"
        ).get_version(0)
        cls.experiment = cls.project.create_experiment(name="yolo-triple-dataset")
        cls.dataset = cls.client.get_dataset_by_id(
            "01888c1b-cfb6-768b-83dd-2e1c460e79cf"
        )
        cls.experiment.attach_dataset(
            name="train", dataset_version=cls.dataset.get_version("train")
        )
        cls.experiment.log_parameters({"epochs": 1, "patience": 500, "image_size": 640})
        cls.experiment.attach_model_version(
            model_version=cls.model_version, do_attach_base_parameters=False
        )

        os.environ["experiment_id"] = str(cls.experiment.id)

        cls.training_pipeline = Yolov8ClassificationTrainer()

    @classmethod
    def tearDownClass(cls) -> None:
        pass
        cls.project.delete()
        shutil.rmtree("data")
        shutil.rmtree("images")
        shutil.rmtree("train")
        os.remove("data.zip")
        os.remove("last.pt")
        os.remove("yolov8x-cls.pt")

    def test_preprocess_train_test(self):
        self.training_pipeline.prepare_data_for_training()
        self.assertTrue(
            len(os.listdir(self.training_pipeline.val_folder_path)),
            len(self.training_pipeline.labelmap),
        )

        self.training_pipeline.train()
        self.assertTrue(os.path.isfile(self.training_pipeline.weights_path))
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.training_pipeline.weights_dir_path, "last.onnx")
            )
        )

        self.training_pipeline.test()
        self.assertTrue(
            type(self.training_pipeline.experiment.get_log("confusion").data) == dict
        )

        self.training_pipeline.eval()
        evaluation_results = self.client.connexion.get(
            f"/sdk/experiment/{str(self.experiment.id)}/evaluations"
        ).json()
        self.assertGreater(evaluation_results["count"], 0)


if __name__ == "__main__":
    unittest.main()
