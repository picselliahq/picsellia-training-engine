import uuid
from typing import List

import pytest
from unittest.mock import patch

from src import Pipeline
from src import pipeline as pipeline_decorator
from src.enums import PipelineState, StepState
from src.models.steps.step_metadata import StepMetadata
from tests.decorators.fixtures.pipeline_fixtures import entrypoint_call_tracker


class TestPipelineDecorator:
    def test_pipeline_decorator_returns_configured_pipeline_object(
        self,
        mock_pipeline_entrypoint,
        mock_pipeline_context,
        mock_pipeline_name,
        mock_pipeline_log_folder_path,
        mock_pipeline_remove_logs_on_completion,
    ):
        decorated_function = pipeline_decorator(
            context=mock_pipeline_context,
            name=mock_pipeline_name,
            log_folder_path=mock_pipeline_log_folder_path,
            remove_logs_on_completion=mock_pipeline_remove_logs_on_completion,
        )(mock_pipeline_entrypoint)

        assert isinstance(
            decorated_function, Pipeline
        ), "The decorated function should be an instance of Pipeline."

        assert (
            decorated_function._context == mock_pipeline_context
        ), "The Pipeline context was not set correctly."
        assert (
            decorated_function.name == mock_pipeline_name
        ), "The Pipeline name was not set correctly."
        assert (
            decorated_function.logger_manager.log_folder_root_path
            == mock_pipeline_log_folder_path
        ), "The log folder path was not set correctly."
        assert (
            decorated_function.remove_logs_on_completion
            is mock_pipeline_remove_logs_on_completion
        ), "The remove_logs_on_completion flag was not set correctly."
        assert (
            decorated_function.entrypoint == mock_pipeline_entrypoint
        ), "The entrypoint was not set to the decorated function correctly."

    def test_pipeline_inside_pipeline_exception(self, mock_pipeline: Pipeline):
        with pytest.raises(RuntimeError):
            with mock_pipeline:
                with Pipeline(
                    context={},
                    name="NestedPipeline",
                    log_folder_path=None,
                    remove_logs_on_completion=True,
                    entrypoint=lambda: None,
                ):
                    pass

    def test_get_context_without_active_pipeline_exception(self):
        with pytest.raises(RuntimeError):
            Pipeline.get_active_context()

    def test_pipeline_cannot_scan_entrypoint(self, mock_pipeline: Pipeline):
        with pytest.raises(ValueError):
            mock_pipeline.entrypoint = None
            mock_pipeline._scan_steps()

    def test_no_active_pipeline_get_active_context(self):
        with pytest.raises(RuntimeError):
            Pipeline.get_active_context()

    def test_active_pipeline_no_context_get_active_context(
        self, mock_pipeline: Pipeline
    ):
        mock_pipeline._context = None
        Pipeline.ACTIVE_PIPELINE = mock_pipeline
        with pytest.raises(RuntimeError):
            Pipeline.get_active_context()

    def test_active_pipeline_get_active_context(self, mock_pipeline: Pipeline):
        Pipeline.ACTIVE_PIPELINE = mock_pipeline
        assert Pipeline.get_active_context() == mock_pipeline._context

    def test_registered_step_metadata_name_in_registry(
        self, mock_step_metadata_1: StepMetadata, mock_step_metadata_name_1: str
    ):
        assert len(Pipeline.STEPS_REGISTRY) == 0
        Pipeline.register_step_metadata(mock_step_metadata_1)

        assert len(Pipeline.STEPS_REGISTRY) == 1
        assert mock_step_metadata_name_1 in Pipeline.STEPS_REGISTRY

    def test_pipeline_state_with_no_steps(self, mock_pipeline):
        assert mock_pipeline.state == PipelineState.PENDING

    def test_execution_calls_entrypoint_only_once(self, mock_pipeline):
        mock_pipeline()
        assert entrypoint_call_tracker["count"] == 1

    def test_pipeline_entrypoint_arguments(self, mock_pipeline):
        mock_pipeline("test", var1=50, var2=60)
        assert len(entrypoint_call_tracker["args"]) == 1
        assert len(entrypoint_call_tracker["kwargs"]) == 2

    @patch("src.decorators.pipeline_decorator.inspect.getsource")
    def test_analyze_and_register_steps(
        self,
        mock_getsource,
        mock_pipeline_entrypoint_source,
        mock_pipeline,
        mock_step_metadata_1,
        mock_step_metadata_2,
    ):
        mock_getsource.return_value = mock_pipeline_entrypoint_source

        # Register dummy steps to simulate finding them during analysis
        Pipeline.STEPS_REGISTRY["step1"] = mock_step_metadata_1
        Pipeline.STEPS_REGISTRY["step2"] = mock_step_metadata_2

        mock_pipeline._scan_steps()

        assert len(mock_pipeline.steps_metadata) == 2
        assert mock_pipeline.steps_metadata[0].name == mock_step_metadata_1.name
        assert mock_pipeline.steps_metadata[1].name == mock_step_metadata_2.name

    def test_log_pipeline_info(self, mock_pipeline):
        with patch.object(mock_pipeline.logger_manager.logger, "info") as mock_info:
            mock_pipeline.log_pipeline_info("Test log")
            mock_info.assert_called_with("Test log")

    def test_log_pipeline_warning(self, mock_pipeline):
        with patch.object(mock_pipeline.logger_manager.logger, "warning") as mock_info:
            mock_pipeline.log_pipeline_warning("Test log")
            mock_info.assert_called_with("Test log")

    def test_finalize_initialization_sets_state_to_running(self, mock_pipeline):
        mock_pipeline()
        assert mock_pipeline._state == PipelineState.RUNNING
        assert mock_pipeline.initialization_log_file_path is not None

    @pytest.mark.parametrize(
        "step_states,expected_state",
        [
            (
                [StepState.SUCCESS, StepState.SUCCESS, StepState.SUCCESS],
                PipelineState.SUCCESS,
            ),
            (
                [StepState.SUCCESS, StepState.RUNNING, StepState.PENDING],
                PipelineState.RUNNING,
            ),
            (
                [StepState.PENDING, StepState.PENDING, StepState.PENDING],
                PipelineState.PENDING,
            ),
            (
                [StepState.SUCCESS, StepState.FAILED, StepState.SKIPPED],
                PipelineState.FAILED,
            ),
            (
                [StepState.SUCCESS, StepState.FAILED, StepState.SUCCESS],
                PipelineState.PARTIAL_SUCCESS,
            ),
        ],
    )
    def test_state_transitions(self, mock_pipeline, step_states, expected_state):
        for i, state in enumerate(step_states, start=1):
            step_name = f"step{i}"
            step_metadata = StepMetadata(
                id=uuid.uuid4(), name=f"step{i}", display_name=step_name, state=state
            )
            mock_pipeline.register_active_step_metadata(step_metadata)

        assert mock_pipeline.state == expected_state

    def test_duplicate_step_registration(self, mock_step_metadata_1):
        Pipeline.register_step_metadata(mock_step_metadata_1)
        with pytest.raises(ValueError):
            Pipeline.register_step_metadata(mock_step_metadata_1)

    def test_pipeline_registered_steps_metadata(
        self, mock_pipeline, mock_step_metadata_1, mock_step_metadata_2
    ):
        mock_pipeline.register_active_step_metadata(mock_step_metadata_1)
        mock_pipeline.register_active_step_metadata(mock_step_metadata_2)
        assert mock_pipeline.steps_metadata == [
            mock_step_metadata_1,
            mock_step_metadata_2,
        ], "Pipeline steps metadata not set correctly."

    @pytest.mark.parametrize(
        "category,context,expected_headers",
        [
            (
                "hyperparameters",
                {"key1": "value1", "key2": "value2"},
                ["hyperparameters", "values"],
            ),
            (
                "augmentation_parameters",
                {"key1": {"subkey": "subvalue"}, "key2": "value2"},
                ["parameters", "values"],
            ),
        ],
    )
    def test_get_markdown_table_headers(
        self,
        mock_pipeline: Pipeline,
        category: str,
        context: dict,
        expected_headers: List[str],
    ):
        tabulate_array = mock_pipeline._compute_markdown_table(category, context)
        headers_line = tabulate_array.split("\n")[0]

        assert isinstance(tabulate_array, str)
        assert all(
            header in headers_line for header in expected_headers
        ), f"Headers {expected_headers} not found in {headers_line}"

    @pytest.mark.parametrize(
        "input_context, expected_flat, expected_nested",
        [
            # Test case with mixed flat and nested parameters
            (
                {"flat1": "value1", "nested": {"nested_key1": "nested_value1"}},
                {"flat1": "value1"},
                {"nested": {"nested_key1": "nested_value1"}},
            ),
            # Test case with only flat parameters
            ({"flat1": "value1", "flat2": 2}, {"flat1": "value1", "flat2": 2}, {}),
            # Test case with only nested parameters
            (
                {
                    "nested1": {"nested_key1": "nested_value1"},
                    "nested2": {"nested_key2": 2},
                },
                {},
                {
                    "nested1": {"nested_key1": "nested_value1"},
                    "nested2": {"nested_key2": 2},
                },
            ),
            # Test case with empty context
            ({}, {}, {}),
        ],
    )
    def test_extract_parameters_context_dict(
        self, input_context, expected_flat, expected_nested
    ):
        pipeline = Pipeline(
            context={},
            name="MyPipeline",
            log_folder_path=None,
            remove_logs_on_completion=True,
            entrypoint=lambda: None,
        )
        (
            flat_parameters,
            nested_parameters,
        ) = pipeline._extract_parameters_from_context_dict(input_context)

        assert (
            flat_parameters == expected_flat
        ), f"Expected flat parameters to be {expected_flat}, got {flat_parameters}"
        assert (
            nested_parameters == expected_nested
        ), f"Expected nested parameters to be {expected_nested}, got {nested_parameters}"

    # Test case for an object with a to_dict method
    def test_parse_context_to_dict_with_object(self, mock_pipeline):
        class ContextWithToDict:
            def to_dict(self):
                return {"key": "value"}

        context = ContextWithToDict()
        assert mock_pipeline._parse_context_to_dict(context) == {
            "key": "value"
        }, "Failed to parse context from object with to_dict method."

    # Test case for a dictionary input
    def test_parse_context_to_dict_with_dict(self, mock_pipeline):
        context = {"key": "value"}
        assert (
            mock_pipeline._parse_context_to_dict(context) == context
        ), "Failed to return the original dictionary context."

    # Test case for inputs that are neither dictionaries nor have a to_dict method
    @pytest.mark.parametrize("context", [None, 42, "string", [1, 2, 3], set()])
    def test_parse_context_to_dict_with_other(self, mock_pipeline, context):
        assert (
            mock_pipeline._parse_context_to_dict(context) is None
        ), f"Expected None for context of type {type(context)}, but didn't get it."

    @pytest.mark.parametrize(
        "context,expected_call_count",
        [
            # Case 1: Context with only flat parameters
            ({"param1": "value1", "param2": "value2"}, 2),
            # Case 2: Context with nested parameters
            ({"param1": "value1", "nestedParam": {"subParam": "subValue"}}, 3),
            (
                {
                    "param1": "value1",
                    "nestedParam1": {"subParam": "subValue"},
                    "nestedParam2": {"subParam": "subValue"},
                },
                4,
            ),
            (
                {
                    "param1": "value1",
                    "nestedParam1": {"subParam1": {"subParam3": "subValue"}},
                    "nestedParam2": {"subParam2": "subValue"},
                },
                4,
            ),
        ],
    )
    def test_log_pipeline_context_call_count(
        self, mock_pipeline, context, expected_call_count
    ):
        with patch.object(mock_pipeline, "log_pipeline_info") as mock_log:
            mock_pipeline._context = context
            mock_pipeline._log_pipeline_context()
            # The expected call count is:
            # 1 for the initial "Pipeline ... is starting with the following context:" message,
            # Plus 1 for each Markdown table generated (1 for flat parameters, and 1 for each nested parameter group)
            assert (
                mock_log.call_count == expected_call_count
            ), f"Expected log_pipeline_info to be called {expected_call_count} times, but got {mock_log.call_count}."

    def test_log_pipeline_no_context_prints_warning(self, mock_pipeline):
        with patch.object(mock_pipeline, "log_pipeline_warning") as mock_log:
            mock_pipeline._context = None
            mock_pipeline._log_pipeline_context()
            mock_log.assert_called_once()
