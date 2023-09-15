import os
import segmentation_models as sm
import keras
from picsellia.exceptions import ResourceNotFoundError


from abstract_trainer.trainer import AbstractTrainer
from utils import (
    split_train_test_val_filenames,
    makedirs_images_masks,
    move_images_and_masks_to_directories,
    Dataset,
    get_training_augmentation,
    get_preprocessing,
    get_validation_augmentation,
    Dataloader,
    format_and_log_eval_metrics,
    download_image_mask_assets,
    get_classes_from_mask_dataset,
    log_training_sample_to_picsellia,
)


class UnetSegmentationTrainer(AbstractTrainer):
    def __init__(self):
        super().__init__()

        self.mask_files = None
        self.image_files = None
        self.image_path = os.path.join(self.experiment.png_dir, "original")
        self.mask_path = os.path.join(self.experiment.png_dir, "masks")

        self.x_train_dir = os.path.join(self.experiment.png_dir, "train-images")
        self.y_train_dir = os.path.join(self.experiment.png_dir, "train-masks")

        self.x_test_dir = os.path.join(self.experiment.png_dir, "test-images")
        self.y_test_dir = os.path.join(self.experiment.png_dir, "test-masks")

        self.x_eval_dir = os.path.join(self.experiment.png_dir, "eval-images")
        self.y_eval_dir = os.path.join(self.experiment.png_dir, "eval-masks")

        self.backbone = "efficientnetb1"
        self.parameters = self.experiment.get_log("parameters").data
        self.preprocess_input = sm.get_preprocessing(self.backbone)
        self.classes = get_classes_from_mask_dataset(self.experiment)
        self.n_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        self.batch_size = int(self.parameters.get("batch_size", 8))
        self.best_model_path = os.path.join(
            self.experiment.checkpoint_dir, "best_model.h5"
        )
        self.metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
        ]

        self.train_dataloader = None
        self.test_dataloader = None
        self.eval_dataloader = None
        self.model = None
        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.best_model_path,
                save_weights_only=True,
                save_best_only=True,
                mode="min",
            ),
            keras.callbacks.ReduceLROnPlateau(),
        ]

    def prepare_data_for_training(self):
        self._download_data()
        self._split_and_move_data()
        self._create_train_test_eval_dataloaders()

    def _download_data(self):
        self.image_files, self.mask_files = download_image_mask_assets(
            self.experiment, self.image_path, self.mask_path
        )

    def _split_and_move_data(self):
        (
            self.train_images_filenames,
            self.test_images_filenames,
            self.eval_images_filenames,
        ) = split_train_test_val_filenames(image_files=self.image_files, seed=11)
        makedirs_images_masks(
            x_train_dir=self.x_train_dir,
            y_train_dir=self.y_train_dir,
            x_test_dir=self.x_test_dir,
            y_test_dir=self.y_test_dir,
            x_eval_dir=self.x_eval_dir,
            y_eval_dir=self.y_eval_dir,
        )
        self._move_all_images_masks_to_directories()

    def _move_all_images_masks_to_directories(self):
        dataset_mappings = [
            (self.train_images_filenames, self.x_train_dir, self.y_train_dir),
            (self.test_images_filenames, self.x_test_dir, self.y_test_dir),
            (self.eval_images_filenames, self.x_eval_dir, self.y_eval_dir),
        ]

        for image_list, dest_image_dir, dest_mask_dir in dataset_mappings:
            move_images_and_masks_to_directories(
                image_path=self.image_path,
                mask_path=self.mask_path,
                image_list=image_list,
                mask_list=self.mask_files,
                dest_image_dir=dest_image_dir,
                dest_mask_dir=dest_mask_dir,
            )

    def _create_train_test_eval_dataloaders(self):
        train_dataset = Dataset(
            self.x_train_dir,
            self.y_train_dir,
            classes=self.classes,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(self.preprocess_input),
        )
        test_dataset = Dataset(
            self.x_test_dir,
            self.y_test_dir,
            classes=self.classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocess_input),
        )
        eval_dataset = Dataset(
            self.x_eval_dir,
            self.y_eval_dir,
            classes=self.classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocess_input),
        )
        log_training_sample_to_picsellia(
            dataset=train_dataset,
            experiment=self.experiment,
        )
        self.train_dataloader = Dataloader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)
        self.eval_dataloader = Dataloader(eval_dataset, batch_size=1, shuffle=False)

    def train(self):
        epochs = int(self.parameters.get("epochs", 1))
        learning_rate = self.parameters.get("learning_rate", 5e-4)
        activation = "sigmoid" if self.n_classes == 1 else "softmax"

        (
            loaded_checkpoint_path,
            loaded_finetuned_checkpoint_path,
        ) = self._download_either_original_or_finetuned_artifact()

        if loaded_checkpoint_path and not loaded_finetuned_checkpoint_path:
            self.model = sm.Unet(
                self.backbone,
                classes=self.n_classes,
                activation=activation,
                encoder_weights=loaded_checkpoint_path,
            )
        elif loaded_finetuned_checkpoint_path and not loaded_checkpoint_path:
            self.model = sm.Unet(
                self.backbone, classes=self.n_classes, activation=activation
            )
            self.model.load_weights(loaded_finetuned_checkpoint_path)

        optim = keras.optimizers.Adam(learning_rate)
        total_loss = self._get_total_loss()
        self.model.compile(optim, total_loss, self.metrics)

        _ = self.model.fit(
            self.train_dataloader,
            steps_per_epoch=len(self.train_dataloader),
            epochs=epochs,
            callbacks=self.callbacks,
            validation_data=self.test_dataloader,
            validation_steps=len(self.test_dataloader),
        )
        self.experiment.store("finetuned_model_weights", self.best_model_path)

    def _download_either_original_or_finetuned_artifact(self):
        try:
            checkpoint_file = self.experiment.get_artifact("model_weights")
            loaded_checkpoint_path = os.path.join(
                self.experiment.checkpoint_dir, checkpoint_file.filename
            )
            checkpoint_file.download(self.experiment.checkpoint_dir)

        except ResourceNotFoundError:
            loaded_checkpoint_path = None

        try:
            finetuned_checkpoint_file = self.experiment.get_artifact(
                "finetuned_model_weights"
            )
            loaded_finetuned_checkpoint_path = os.path.join(
                self.experiment.checkpoint_dir,
                "finetuned",
                finetuned_checkpoint_file.filename,
            )
            finetuned_checkpoint_file.download(
                os.path.join(self.experiment.checkpoint_dir, "finetuned")
            )
            loaded_checkpoint_path = None

        except ResourceNotFoundError:
            loaded_finetuned_checkpoint_path = None

        return loaded_checkpoint_path, loaded_finetuned_checkpoint_path

    def _get_total_loss(self) -> sm.base.objects.SumOfLosses:
        dice_loss = sm.losses.DiceLoss()
        focal_loss = (
            sm.losses.BinaryFocalLoss()
            if self.n_classes == 1
            else sm.losses.CategoricalFocalLoss()
        )
        total_loss = dice_loss + (1 * focal_loss)
        return total_loss

    def eval(self):
        self.model.load_weights(self.best_model_path)
        scores = self.model.evaluate(self.eval_dataloader)
        format_and_log_eval_metrics(self.experiment, self.metrics, scores)
