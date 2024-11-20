import os

from src import Pipeline, step
from src.models.model.common.model_context import ModelContext


@step
def model_version_weights_extractor():
    context = Pipeline.get_active_context()

    model_version = context.client.get_model_version_by_id(id=context.model_version_id)
    model_context = ModelContext(
        model_name=model_version.name,
        model_version=model_version,
        labelmap=model_version.labels,
    )

    model_context.download_weights(
        destination_path=os.path.join(os.getcwd(), str(model_version.id))
    )
    return model_context
