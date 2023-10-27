import os
import shutil
import time
import unittest
from datetime import date

import numpy as np
from picsellia import Client

from utils import (
    get_classes_from_mask_dataset,
    download_image_mask_assets,
    get_image_mask_assets,
    split_train_test_val_filenames,
    makedirs_images_masks,
    _find_mask_by_image,
    _change_mask_filename_to_match_image,
    get_mask_file_extension,
    extract_classes_from_mask,
    Dataset,
    Dataloader,
)
from trainer import UnetSegmentationTrainer
from picsellia.sdk.dataset import DatasetVersion

TOKEN = os.environ["api_token"]
ORGA_ID = os.environ["organization_id"]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestUnetSegmentation(unittest.TestCase):
    organization_id = None
    model_version = None
    dataset = None
    experiment = None
    project = None
    client = None
    token: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.token = TOKEN
        cls.organization_id = ORGA_ID
        cls.client = Client(
            api_token=cls.token,
            organization_id=cls.organization_id,
            host="https://staging.picsellia.com/",
        )
        cls.project = cls.client.create_project(
            name=f"test_unet{str(date.today())}-{str(time.time())}"
        )
        cls.model_version = cls.client.get_model_version_by_id(
            "018a99a0-0b68-772e-a04d-47e0c7096e66"
        )

        cls.experiment = cls.project.create_experiment(name="car-segmentation-unet")
        cls.dataset = cls.client.get_dataset_by_id(
            "018a89c6-0bc8-7de8-8ba2-487fd24567ad"
        )

        cls.experiment.attach_dataset(
            name="images", dataset_version=cls.dataset.get_version("images")
        )
        cls.experiment.attach_dataset(
            name="masks", dataset_version=cls.dataset.get_version("masks")
        )
        cls.model_version = cls.client.get_model_version_by_id(
            "018a703e-80fc-7d8f-9983-1430e5f7a427"
        )
        cls.experiment.log_parameters(
            {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "mask_filename_prefix": "mask",
                "image_filename_prefix": "orig",
            }
        )
        cls.experiment.attach_model_version(
            model_version=cls.model_version, do_attach_base_parameters=False
        )

        cls.image_path = os.path.join(cls.experiment.png_dir, "original")
        cls.mask_path = os.path.join(cls.experiment.png_dir, "masks")

        cls.x_train_dir = os.path.join(cls.experiment.png_dir, "train-images")
        cls.y_train_dir = os.path.join(cls.experiment.png_dir, "train-masks")

        cls.x_test_dir = os.path.join(cls.experiment.png_dir, "test-images")
        cls.y_test_dir = os.path.join(cls.experiment.png_dir, "test-masks")

        cls.x_eval_dir = os.path.join(cls.experiment.png_dir, "eval-images")
        cls.y_eval_dir = os.path.join(cls.experiment.png_dir, "eval-masks")

        cls.parameters = cls.experiment.get_log("parameters").data
        cls.mask_prefix = cls.parameters["mask_filename_prefix"]
        cls.image_prefix = cls.parameters["image_filename_prefix"]

        os.environ["experiment_id"] = str(cls.experiment.id)
        cls.training_pipeline = UnetSegmentationTrainer()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()
        if os.path.isdir(cls.experiment.name):
            shutil.rmtree(cls.experiment.name)

    def test_get_classes_from_mask_dataset(self):
        expected_results = ["car"]
        results = get_classes_from_mask_dataset(self.experiment)
        self.assertEqual(expected_results, results)

    def test_download_image_mask_assets(self):
        image_files, mask_files = download_image_mask_assets(
            experiment=self.experiment,
            image_path=self.image_path,
            mask_path=self.mask_path,
        )
        self.assertNotEquals((image_files, mask_files), ([], []))
        self.assertTrue(os.path.exists(self.image_path))
        self.assertTrue(os.path.exists(self.mask_path))

    def test_get_image_mask_assets(self):
        image_assets, mask_assets = get_image_mask_assets(
            experiment=self.experiment,
            dataset_list=self.experiment.list_attached_dataset_versions(),
        )

        self.assertEqual(
            (type(image_assets), type(mask_assets)), (DatasetVersion, DatasetVersion)
        )

    def test_split_train_test_val_filenames(self):
        image_files = [
            "orig - DS834.JPG",
            "orig - DS1058.jpg",
            "orig - DS500.jpg",
            "orig - DS1258.JPG",
            "orig - DS758.JPG",
            "orig - DS559.JPG",
            "orig - DS420.jpg",
            "orig - DS1054.JPG",
            "orig - DS1290.jpg",
        ]

        (
            train_image_filenames,
            test_images_filenames,
            eval_images_filenames,
        ) = split_train_test_val_filenames(image_files=image_files, seed=11)

        self.assertEqual(len(train_image_filenames), 7)
        self.assertEqual(len(test_images_filenames), 1)
        self.assertEqual(len(eval_images_filenames), 1)

    def test_makedirs_images_masks(self):
        all_directories_are_created = True
        makedirs_images_masks(
            x_train_dir=self.x_train_dir,
            y_train_dir=self.y_train_dir,
            x_test_dir=self.x_test_dir,
            y_test_dir=self.y_test_dir,
            x_eval_dir=self.x_eval_dir,
            y_eval_dir=self.y_eval_dir,
        )
        for directory_path in [
            self.x_train_dir,
            self.y_test_dir,
            self.x_test_dir,
            self.y_test_dir,
            self.y_eval_dir,
            self.x_eval_dir,
        ]:
            if not os.path.exists(directory_path):
                all_directories_are_created = False

        self.assertTrue(all_directories_are_created)

    def test_find_mask_by_image(self):
        mask_files = [
            "mask - DS944.png",
            "mask - DS1238.png",
            "mask - DS363.png",
            "mask - DS982.png",
        ]
        resulting_mask_file = _find_mask_by_image(
            image_filename="orig - DS1238.jpg",
            mask_files=mask_files,
            image_prefix=self.image_prefix,
            mask_prefix=self.mask_prefix,
        )
        expected_mask_file = "mask - DS1238.png"
        self.assertEqual(expected_mask_file, resulting_mask_file)

    def test_find_mask_by_image_not_found(self):
        mask_files = [
            "mask - DS944.png",
            "mask - DS363.png",
            "mask - DS982.png",
        ]
        image_filename = "orig - DS1238.jpg"
        with self.assertRaises(ValueError) as context:
            _find_mask_by_image(
                image_filename=image_filename,
                mask_files=mask_files,
                image_prefix=self.image_prefix,
                mask_prefix=self.mask_prefix,
            )

        self.assertIn("No mask found for image", str(context.exception))

    def test_change_mask_filename_to_match_image(self):
        new_mask_filename = _change_mask_filename_to_match_image(
            mask_prefix="mask",
            image_prefix="orig",
            old_mask_filename="mask - DS944.png",
        )
        expected_mask_filename = "orig - DS944.png"

        self.assertEqual(expected_mask_filename, new_mask_filename)

    def test_get_mask_file_extension(self):
        file_extension = get_mask_file_extension(mask_file_path="home/mask - DS944.png")
        self.assertEqual("png", file_extension)

    def test_extract_classes_from_mask(self):
        mask = np.array([[1, 1, 2], [2, 3, 3]])
        class_values = [1]
        expected_result = np.array([[[1.0], [1.0], [0.0]], [[0.0], [0.0], [0.0]]])
        result = extract_classes_from_mask(mask, class_values)

        self.assertIsNone(np.testing.assert_array_equal(expected_result, result))

    def test_launch_training_pipeline(self):
        self.training_pipeline.prepare_data_for_training()
        directories_to_check = [
            self.training_pipeline.x_train_dir,
            self.training_pipeline.y_train_dir,
            self.training_pipeline.x_test_dir,
            self.training_pipeline.y_test_dir,
            self.training_pipeline.x_eval_dir,
            self.training_pipeline.y_eval_dir,
        ]
        for directory in directories_to_check:
            self.assertGreaterEqual(len(os.listdir(directory)), 1)
        self.assertNotEquals(self.training_pipeline.train_dataloader, None)
        self.assertNotEquals(self.training_pipeline.test_dataloader, None)
        self.assertNotEquals(self.training_pipeline.eval_dataloader, None)

        self.training_pipeline.train()
        self.assertTrue(os.path.isfile(self.training_pipeline.best_model_path))

        self.training_pipeline.eval()
        self.assertNotEquals(
            type(self.training_pipeline.experiment.get_log(name="eval-results").data),
            None,
        )


class TestDataset(unittest.TestCase):
    preprocessing = None
    augmentation = None
    masks_dir = None
    images_dir = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.images_dir = "test_files/images"
        cls.masks_dir = "test_files/masks"
        classes = ["class1"]
        cls.dataset = Dataset(
            cls.images_dir,
            cls.masks_dir,
            classes,
            cls.augmentation,
            cls.preprocessing,
        )

    def test_dataset_constructor(self):
        self.assertEqual(self.dataset.augmentation, self.augmentation)
        self.assertEqual(self.dataset.preprocessing, self.preprocessing)
        self.assertEqual(len(self.dataset), len(os.listdir(self.images_dir)))

    def test_get_item_dataset(self):
        index = 0
        image, mask = self.dataset[index]
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(image.shape[-1], 3)
        self.assertEqual(mask.shape[-1], 1)

    def test_get_item_dataset_multiclass(self):
        classes = ["class1", "class2"]
        augmentation = None
        preprocessing = None
        dataset = Dataset(
            self.images_dir, self.masks_dir, classes, augmentation, preprocessing
        )
        index = 0
        image, mask = dataset[index]
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(image.shape[-1], 3)
        self.assertEqual(mask.shape[-1], len(classes) + 1)

    def test_len_dataset(self):
        classes = ["class1", "class2"]
        augmentation = None
        preprocessing = None
        dataset = Dataset(
            self.images_dir, self.masks_dir, classes, augmentation, preprocessing
        )
        expected_length = len(os.listdir(self.images_dir))
        actual_length = len(dataset)

        self.assertEqual(actual_length, expected_length)


class TestDataloader(unittest.TestCase):
    batch_size = 4
    shuffle = True

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = Dataset("test_files/images", "test_files/masks", ["class1"])

    def test_dataloader_constructor(self, batch_size=None, shuffle=True):
        dataloader = Dataloader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.assertEqual(dataloader.batch_size, batch_size)
        self.assertEqual(dataloader.shuffle, shuffle)

    def test_get_item_dataloader(self):
        batch_size = 2
        dataloader = Dataloader(self.dataset, batch_size=batch_size)
        index = 0
        batch = dataloader[index]
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 2)  # Check the batch size
        self.assertIsInstance(batch[0], np.ndarray)
        self.assertIsInstance(batch[1], np.ndarray)

    def test_get_len_dataloader(self):
        batch_size = 2
        dataloader = Dataloader(self.dataset, batch_size=batch_size)
        expected_length = len(self.dataset) // batch_size
        actual_length = len(dataloader)

        self.assertEqual(expected_length, actual_length)


if __name__ == "__main__":
    unittest.main()
