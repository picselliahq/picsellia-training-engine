import os
import random
import shutil

import albumentations as A
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from picsellia import Experiment
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.dataset import DatasetVersion
from picsellia.types.enums import LogType

SIZE = 640


def get_classes_from_mask_dataset(experiment: Experiment) -> list[str]:
    mask_dataset = experiment.get_dataset(name="masks")
    labels = mask_dataset.list_labels()

    return [label.name for label in labels]


def download_image_mask_assets(
    experiment: Experiment, image_path: str, mask_path: str
) -> tuple[list[str], list[str]]:
    image_assets, mask_assets = get_image_mask_assets(
        experiment, experiment.list_attached_dataset_versions()
    )
    image_assets.download(target_path=image_path)
    mask_assets.download(target_path=mask_path)
    image_files = os.listdir(path=image_path)
    mask_files = os.listdir(path=mask_path)

    return image_files, mask_files


def get_image_mask_assets(
    experiment: Experiment, dataset_list: list
) -> tuple[DatasetVersion, DatasetVersion]:
    attached_dataset_names = [
        dataset_version.version for dataset_version in dataset_list
    ]
    if len(attached_dataset_names) != 2:
        raise Exception(
            "You must have exactly two datasets, 'original' for the original images, and 'masks' for the masks "
        )

    try:
        image_assets = experiment.get_dataset(name="original")
    except Exception:
        raise ResourceNotFoundError(
            "Can't find 'original' datasetversion. Expecting 'original' and 'masks', as attached datasets"
        )

    try:
        mask_assets = experiment.get_dataset(name="masks")
    except Exception:
        raise ResourceNotFoundError(
            "Can't find 'masks' datasetversion. Expecting 'original' and 'masks', as attached datasets"
        )

    return image_assets, mask_assets


def split_train_test_val_filenames(
    image_files: list[str], seed: int
) -> tuple[list[str], list[str], list[str]]:
    random.Random(seed).shuffle(image_files)
    nbr_images = len(image_files)
    train_image_filenames, test_images_filenames, eval_images_filenames = np.split(
        image_files, [int(nbr_images * 0.8), int(nbr_images * 0.9)]
    )

    return train_image_filenames, test_images_filenames, eval_images_filenames


def makedirs_images_masks(
    x_train_dir: str,
    y_train_dir: str,
    x_test_dir: str,
    y_test_dir: str,
    x_eval_dir: str,
    y_eval_dir: str,
) -> None:
    os.makedirs(name=x_train_dir)
    os.makedirs(name=y_train_dir)

    os.makedirs(name=x_test_dir)
    os.makedirs(name=y_test_dir)

    os.makedirs(name=x_eval_dir)
    os.makedirs(name=y_eval_dir)


def move_images_and_masks_to_directories(
    image_path: str,
    mask_path: str,
    image_list: list[str],
    mask_list: list[str],
    dest_image_dir: str,
    dest_mask_dir: str,
):
    for image_filename in tqdm.tqdm(image_list):
        try:
            mask_filename = _find_mask_by_image(
                image_filename=image_filename, mask_files=mask_list
            )
        except ValueError:
            continue

        image_dest = os.path.join(dest_image_dir, image_filename)
        mask_dest = os.path.join(
            dest_mask_dir,
            _change_mask_filename_to_match_image(
                mask_prefix="mask", image_prefix="orig", old_mask_filename=mask_filename
            ),
        )

        img_file_path = os.path.join(image_path, image_filename)
        mask_file_path = os.path.join(mask_path, mask_filename)

        shutil.copy(img_file_path, image_dest)
        shutil.copy(mask_file_path, mask_dest)


def _find_mask_by_image(image_filename: str, mask_files: list[str]) -> str:
    base_filename = image_filename.split("- ")[1].split(".")[0]
    for mask_file in mask_files:
        if base_filename in mask_file:
            return mask_file
    raise ValueError(f"No mask found for image {image_filename}")


def _change_mask_filename_to_match_image(
    mask_prefix: str, image_prefix: str, old_mask_filename: str
) -> str:
    new_mask_filename = image_prefix + old_mask_filename[len(mask_prefix) :]

    return new_mask_filename


class Dataset:
    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_filenames = [
            os.path.join(images_dir, image_id) for image_id in self.ids
        ]
        self.masks_filenames = [
            os.path.join(
                masks_dir,
                image_id.split(".")[0]
                + "."
                + get_mask_file_extension(os.listdir(masks_dir)[0]),
            )
            for image_id in self.ids
        ]
        self.class_values = [classes.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.images_filenames[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_filenames[i], 0)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self) -> int:
        return len(self.ids)


def get_mask_file_extension(mask_file_path: str) -> str:
    return mask_file_path.split(".")[1]


def extract_classes_from_mask(mask: np.ndarray, class_values: list) -> np.ndarray:
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype("float")
    return mask


class Dataloader(keras.utils.Sequence):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i) -> list:
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self) -> int:
        number_of_batches_per_epoch = len(self.indexes) // self.batch_size
        return number_of_batches_per_epoch

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def get_training_augmentation() -> A.Compose:
    train_transform = [
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.2, border_mode=0
        ),
        A.Resize(SIZE, SIZE, p=1, always_apply=True),
        A.PadIfNeeded(
            min_height=SIZE, min_width=SIZE, always_apply=True, border_mode=0
        ),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.Sharpen(p=0.6),
                A.Blur(blur_limit=3, p=0.4),
                A.MotionBlur(blur_limit=3, p=0.4),
            ],
            p=0.4,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=0.7),
                A.HueSaturationValue(p=0.7),
            ],
            p=0.4,
        ),
        A.Lambda(mask=round_clip_0_1),
    ]
    return A.Compose(train_transform, is_check_shapes=False)


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_validation_augmentation() -> A.Compose:
    test_transform = [
        A.PadIfNeeded(SIZE, SIZE),
        A.Resize(SIZE, SIZE, p=1, always_apply=True),
    ]
    return A.Compose(test_transform, is_check_shapes=False)


def get_preprocessing(preprocessing_fn) -> A.Compose:
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def format_and_log_eval_metrics(experiment: Experiment, metrics: list, scores: list):
    eval_metrics = {"loss": float("{:.5f}".format(scores[0]))}
    for metric, value in zip(metrics, scores[1:]):
        eval_metrics[metric.__name__] = float("{:.5f}".format(value))

    experiment.log(name="eval-results", type=LogType.TABLE, data=eval_metrics)


def save_training_sample_file(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    output_path = "fig.jpg"
    plt.savefig(output_path)
    return output_path


def log_image_to_picsellia(
    file_path_to_log: str, experiment: Experiment, log_name: str
):
    experiment.log(log_name, type=LogType.IMAGE, data=file_path_to_log)


def log_training_sample_to_picsellia(dataset: Dataset, experiment: Experiment):
    image_index_to_log = 0
    image_to_log, mask_to_log = dataset[image_index_to_log]
    output_path = save_training_sample_file(
        image=image_to_log, mask=mask_to_log[..., 0].squeeze()
    )
    log_image_to_picsellia(
        file_path_to_log=output_path,
        experiment=experiment,
        log_name=f"sample-{str(image_index_to_log)}",
    )
