from src import pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.parameters.processing.processing_model_version_converter_parameters import (
    ProcessingYoloXModelVersionConverterParameters,
)
from src.steps.model_loading.common.yolox_model_context_loader import (
    yolox_model_context_loader,
)
from src.steps.processing.model_version_convertion.model_version_converter_processing import (
    yolox_model_version_converter_processing,
)
from src.steps.weights_extraction.common.model_version_weights_extractor import (
    model_version_weights_extractor,
)


def get_context() -> (
    PicselliaProcessingContext[ProcessingYoloXModelVersionConverterParameters]
):
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingYoloXModelVersionConverterParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def model_version_converter_pipeline() -> None:
    model_context = model_version_weights_extractor()
    model_context = yolox_model_context_loader(model_context=model_context)

    yolox_model_version_converter_processing(model_context=model_context)


if __name__ == "__main__":
    model_version_converter_pipeline()
