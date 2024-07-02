from src.models.parameters.processing.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)


class TestProcessingDiversifiedDataExtractorParameters:
    def test_diversified_data_extractor_parameters_default_values(self):
        # Test that default values are used when no data is provided
        params = ProcessingDiversifiedDataExtractorParameters(log_data={})

        assert params.distance_threshold == 5, "Default distance_threshold should be 5."
        assert (
            params.embedding_model == "openclip"
        ), "Default embedding_model should be 'openclip'."
        assert (
            params.model_architecture == "ViT-B-16-plus-240"
        ), "Default model_architecture should be 'ViT-B-16-plus-240'."
        assert (
            params.pretrained_weights == "laion400m_e32"
        ), "Default pretrained_weights should be 'laion400m_e32'."

    def test_diversified_data_extractor_parameters_extraction_by_keys(self):
        log_data = {
            "dist": 10,
            "model": "custom_model",
            "architecture": "Custom-Arch",
            "weights": "custom_weights",
        }
        params = ProcessingDiversifiedDataExtractorParameters(log_data)

        assert (
            params.distance_threshold == 10
        ), "distance_threshold should have been extracted using 'dist'."
        assert (
            params.embedding_model == "custom_model"
        ), "embedding_model should have been extracted using 'model'."
        assert (
            params.model_architecture == "Custom-Arch"
        ), "model_architecture should have been extracted using 'architecture'."
        assert (
            params.pretrained_weights == "custom_weights"
        ), "pretrained_weights should have been extracted using 'weights'."
