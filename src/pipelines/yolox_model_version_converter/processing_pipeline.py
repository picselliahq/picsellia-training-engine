from src import pipeline
from src.models.contexts.processing.picsellia_model_processing_context import (
    PicselliaModelProcessingContext,
)
from src.models.parameters.processing.processing_model_version_converter_parameters import (
    ProcessingYoloXModelVersionConverterExportParameters,
    ProcessingYoloXModelVersionConverterParameters,
)
from src.steps.model_export.common.yolox_model_exporter import (
    yolox_model_context_exporter,
)
from src.steps.model_loading.common.yolox_model_context_loader import (
    yolox_model_context_loader,
)
from src.steps.weights_extraction.common.model_version_weights_extractor import (
    model_version_weights_extractor,
)


def get_context() -> (
    PicselliaModelProcessingContext[
        ProcessingYoloXModelVersionConverterParameters,
        ProcessingYoloXModelVersionConverterExportParameters,
    ]
):
    return PicselliaModelProcessingContext(
        processing_parameters_cls=ProcessingYoloXModelVersionConverterParameters,
        export_parameters_cls=ProcessingYoloXModelVersionConverterExportParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def model_version_converter_pipeline() -> None:
    model_context = model_version_weights_extractor()
    model_context = yolox_model_context_loader(model_context=model_context)

    yolox_model_context_exporter(model_context=model_context)


if __name__ == "__main__":
    model_version_converter_pipeline()
