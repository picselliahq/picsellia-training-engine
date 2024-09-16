# type: ignore
from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.paddle_ocr.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from src.models.parameters.training.paddle_ocr.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from src.models.steps.model_training.paddle_ocr_model_collection_trainer import (
    PaddleOCRModelCollectionTrainer,
)


@step
def paddle_ocr_model_collection_trainer(model_collection):
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()
    model_trainer = PaddleOCRModelCollectionTrainer(
        model_collection=model_collection, experiment=context.experiment
    )
    model_collection = model_trainer.train_model_collection(
        bbox_epochs=context.hyperparameters.bbox_epochs,
        text_epochs=context.hyperparameters.text_epochs,
    )
    return model_collection
