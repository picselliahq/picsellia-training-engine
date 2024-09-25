import logging
from typing import Union

from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.processing.datalake_collection import DatalakeCollection
from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_prediction.common.miniGPT.minigpt_model_context_predictor import (
    MiniGPTModelContextPredictor,
)


@step
def minigpt_datalake_autotagging_processing(
    datalake: Union[DatalakeContext, DatalakeCollection], model_context: ModelContext
):
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_context_predictor = MiniGPTModelContextPredictor(
        model_context=model_context,
        tags_list=context.processing_parameters.tags_list,
    )
    if isinstance(datalake, DatalakeContext):
        datalake_context = datalake
    elif isinstance(datalake, DatalakeCollection):
        datalake_context = datalake["input"]
    else:
        raise ValueError(
            "Datalake should be either a DatalakeContext or a DatalakeCollection"
        )

    image_paths = model_context_predictor.pre_process_datalake_context(
        datalake_context=datalake_context
    )
    image_batches = model_context_predictor.prepare_batches(
        image_paths=image_paths,
        batch_size=context.processing_parameters.batch_size,
    )
    batch_results = model_context_predictor.run_inference_on_batches(
        image_batches=image_batches
    )
    picsellia_datalake_autotagging_predictions = (
        model_context_predictor.post_process_batches(
            image_batches=image_batches,
            batch_results=batch_results,
            datalake_context=datalake_context,
        )
    )
    print(
        f"picsellia_datalake_autotagging_predictions: {picsellia_datalake_autotagging_predictions}"
    )

    # if isinstance(datalake, DatalakeContext):
    #     datalake_context = datalake
    # elif isinstance(datalake, DatalakeCollection):
    #     datalake_context = datalake["input"]
    # else:
    #     raise ValueError(
    #         "Datalake should be either a DatalakeContext or a DatalakeCollection"
    #     )
    # picsellia_datalake_autotagging_predictions = []
    # for image_name in os.listdir(datalake_context.image_dir):
    #     data = datalake_context.datalake.list_data(
    #         ids=[UUID(image_name.split(".")[0])]
    #     )[0]
    #     picsellia_datalake_autotagging_predictions.append(
    #         {
    #             "data": data,
    #             "tag": [datalake_context.datalake.get_or_create_data_tag(name="text")],
    #         }
    #     )

    logging.info(f"Predictions for datalake {datalake_context.datalake.id} done.")

    for (
        picsellia_datalake_autotagging_prediction
    ) in picsellia_datalake_autotagging_predictions:
        if not picsellia_datalake_autotagging_prediction["tag"]:
            continue
        picsellia_datalake_autotagging_prediction["data"].add_tags(
            tags=picsellia_datalake_autotagging_prediction["tag"]
        )

    logging.info(f"Tags added to datalake {datalake_context.datalake.id}.")
