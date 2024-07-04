from src.models.steps.model_training.paddle_ocr_model_trainer import (
    PaddleOCRModelTrainer,
)


def paddle_ocr_trainer(model_collection):
    model_trainer = PaddleOCRModelTrainer(model_collection=model_collection)

    model_collection = model_trainer.train()

    return model_collection
