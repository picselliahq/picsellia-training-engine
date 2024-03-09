from ultralytics.models.yolo.classify import ClassificationTrainer

from poc.step import step


@step
def model_trainer(trainer: ClassificationTrainer):
    trainer.train()
    return trainer
