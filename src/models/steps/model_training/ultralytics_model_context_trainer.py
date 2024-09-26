from picsellia import Experiment

from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.model.common.model_context import ModelContext
from src.models.model.ultralytics.ultralytics_callbacks import UltralyticsCallbacks
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)

from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)


class UltralyticsModelContextTrainer:
    """
    Trainer class for handling the training process of a model using the Ultralytics framework.

    Attributes:
        model_context (ModelContext): The context containing the model and its associated paths and metadata.
        experiment (Experiment): The experiment instance used for logging and tracking the training process.
        callbacks (UltralyticsCallbacks): The callbacks used during training, initialized based on the provided experiment.
    """

    def __init__(
        self,
        model_context: ModelContext,
        experiment: Experiment,
    ):
        """
        Initializes the trainer with a model context and an experiment.

        Args:
            model_context (ModelContext): The context of the model to be trained.
            experiment (Experiment): The experiment instance used for logging and tracking.
        """
        self.model_context = model_context
        self.experiment = experiment
        self.callback_handler = UltralyticsCallbacks(experiment)

    def _setup_callbacks(self):
        """
        Sets up the callbacks for the model training process.
        """
        for event, callback in self.callback_handler.get_callbacks().items():
            self.model_context.loaded_model.add_callback(event, callback)

    def train_model_context(
        self,
        dataset_collection: TrainingDatasetCollection,
        hyperparameters: UltralyticsHyperParameters,
        augmentation_parameters: UltralyticsAugmentationParameters,
    ) -> ModelContext:
        """
        Trains the model within the provided context using the given datasets, hyperparameters, and augmentation parameters.

        Args:
            dataset_collection (TrainingDatasetCollection): The collection of datasets used for training.
            hyperparameters (UltralyticsHyperParameters): The hyperparameters used for training.
            augmentation_parameters (UltralyticsAugmentationParameters): The augmentation parameters applied during training.

        Returns:
            ModelContext: The updated model context after training.
        """

        self._setup_callbacks()

        if hyperparameters.epochs > 0:
            self.model_context.loaded_model.train(
                # Hyperparameters
                data=dataset_collection.dataset_path,
                epochs=hyperparameters.epochs,
                time=hyperparameters.time,
                patience=hyperparameters.patience,
                batch=hyperparameters.batch_size,
                imgsz=hyperparameters.image_size,
                save=True,
                save_period=hyperparameters.save_period,
                cache=hyperparameters.cache,
                device=hyperparameters.device,
                workers=hyperparameters.workers,
                project=self.model_context.results_dir,
                name=self.model_context.model_name,
                exist_ok=True,
                pretrained=True,
                optimizer=hyperparameters.optimizer,
                seed=hyperparameters.seed,
                deterministic=hyperparameters.deterministic,
                single_cls=hyperparameters.single_cls,
                rect=hyperparameters.rect,
                cos_lr=hyperparameters.cos_lr,
                close_mosaic=hyperparameters.close_mosaic,
                amp=hyperparameters.amp,
                fraction=hyperparameters.fraction,
                profile=hyperparameters.profile,
                freeze=hyperparameters.freeze,
                lr0=hyperparameters.lr0,
                lrf=hyperparameters.lrf,
                momentum=hyperparameters.momentum,
                weight_decay=hyperparameters.weight_decay,
                warmup_epochs=hyperparameters.warmup_epochs,
                warmup_momentum=hyperparameters.warmup_momentum,
                warmup_bias_lr=hyperparameters.warmup_bias_lr,
                box=hyperparameters.box,
                cls=hyperparameters.cls,
                dfl=hyperparameters.dfl,
                pose=hyperparameters.pose,
                kobj=hyperparameters.kobj,
                label_smoothing=hyperparameters.label_smoothing,
                nbs=hyperparameters.nbs,
                overlap_mask=hyperparameters.overlap_mask,
                mask_ratio=hyperparameters.mask_ratio,
                dropout=hyperparameters.dropout,
                val=hyperparameters.validate,
                plots=hyperparameters.plots,
                # Augmentation parameters
                hsv_h=augmentation_parameters.hsv_h,
                hsv_s=augmentation_parameters.hsv_s,
                hsv_v=augmentation_parameters.hsv_v,
                degrees=augmentation_parameters.degrees,
                translate=augmentation_parameters.translate,
                scale=augmentation_parameters.scale,
                shear=augmentation_parameters.shear,
                perspective=augmentation_parameters.perspective,
                flipud=augmentation_parameters.flipud,
                fliplr=augmentation_parameters.fliplr,
                bgr=augmentation_parameters.bgr,
                mosaic=augmentation_parameters.mosaic,
                mixup=augmentation_parameters.mixup,
                copy_paste=augmentation_parameters.copy_paste,
                auto_augment=augmentation_parameters.auto_augment,
                erasing=augmentation_parameters.erasing,
                crop_fraction=augmentation_parameters.crop_fraction,
            )

        return self.model_context
