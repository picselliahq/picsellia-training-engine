from typing import Optional, Dict, Any

from src.models.model.common.model_context import ModelContext

from picsellia import Label, ModelVersion


class HuggingFaceModelContext(ModelContext):
    def __init__(
        self,
        hugging_face_model_name: str,
        model_name: str,
        model_version: ModelVersion,
        pretrained_weights_name: Optional[str] = None,
        trained_weights_name: Optional[str] = None,
        config_name: Optional[str] = None,
        exported_weights_name: Optional[str] = None,
        labelmap: Optional[Dict[str, Label]] = None,
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            pretrained_weights_name=pretrained_weights_name,
            trained_weights_name=trained_weights_name,
            config_name=config_name,
            exported_weights_name=exported_weights_name,
            labelmap=labelmap,
        )
        self.hugging_face_model_name = hugging_face_model_name
        self.processor: Optional[Any] = None
