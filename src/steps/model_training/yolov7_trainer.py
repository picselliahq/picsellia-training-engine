from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.yolov7_dataset_collection import Yolov7DatasetCollection
from src.models.model.yolov7_model_context import Yolov7ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.yolov7.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from src.models.parameters.training.yolov7.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)
from src.models.steps.model_training.common.yolov7_model_context_trainer import (
    Yolov7ModelContextTrainer,
)


@step
def yolov7_model_context_trainer(
    model_context: Yolov7ModelContext, dataset_collection: Yolov7DatasetCollection
) -> Yolov7ModelContext:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_trainer = Yolov7ModelContextTrainer(
        model_context=model_context, experiment=context.experiment
    )

    if (
        not context.api_token
        or not context.organization_id
        or not context.experiment_id
        or not context.host
    ):
        raise ValueError(
            "API token, organization ID, experiment ID, and host must be set"
        )

    model_trainer.train_model_context(
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
        api_token=context.api_token,
        organization_id=context.organization_id,
        host=context.host,
        experiment_id=context.experiment_id,
    )
    
    model_context.set_trained_weights_path()
    
    return model_context
