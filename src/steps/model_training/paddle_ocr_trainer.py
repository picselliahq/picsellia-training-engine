from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.steps.model_training.paddle_ocr_model_trainer import (
    PaddleOCRModelTrainer,
)


@step
def paddle_ocr_trainer(model_collection):
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    model_trainer = PaddleOCRModelTrainer(
        model_collection=model_collection, experiment=context.experiment
    )
    model_collection = model_trainer.train()
    return model_collection
