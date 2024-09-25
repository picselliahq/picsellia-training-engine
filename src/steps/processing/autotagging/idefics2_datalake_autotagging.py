import logging
from typing import Union

from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.processing.datalake_collection import DatalakeCollection
from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_prediction.common.IDEFICS2.idefics2_model_context_predictor import (
    VLMHuggingFaceModelContextPredictor,
)


@step
def idefics2_datalake_autotagging_processing(
    datalake: Union[DatalakeContext, DatalakeCollection], model_context: ModelContext
):
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_context_predictor = VLMHuggingFaceModelContextPredictor(
        model_context=model_context,
        model_name="HuggingFaceM4/idefics2-8b",
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

    image_inputs = model_context_predictor.pre_process_datalake_context(
        datalake_context=datalake_context, device=context.processing_parameters.device
    )
    image_batches = model_context_predictor.prepare_batches(
        image_inputs=image_inputs,
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

    logging.info(f"Predictions for datalake {datalake_context.datalake.id} done.")

    for (
        picsellia_datalake_autotagging_prediction
    ) in picsellia_datalake_autotagging_predictions:
        picsellia_datalake_autotagging_prediction["data"].add_tags(
            tags=picsellia_datalake_autotagging_prediction["tag"]
        )

    logging.info(f"Tags added to datalake {datalake_context.datalake.id}.")
