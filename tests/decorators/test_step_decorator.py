from unittest.mock import patch

import pytest

from src import Pipeline, Step
from src import step as step_decorator
from src.enums import PipelineState, StepState


class TestStepDecorator:
    def test_step_decorator_returns_configured_step_object(
        self,
        mock_step_entrypoint_pass,
        mock_step_metadata_name_1,
        mock_step_continue_on_failure_1,
    ):
        decorated_function = step_decorator(
            name=mock_step_metadata_name_1,
            continue_on_failure=mock_step_continue_on_failure_1,
        )(mock_step_entrypoint_pass)

        assert isinstance(
            decorated_function, Step
        ), "The decorated function should be an instance of Step."

        assert (
            decorated_function.step_name == mock_step_entrypoint_pass.__name__
        ), "The step's name not set correctly."

        assert (
            decorated_function.continue_on_failure == mock_step_continue_on_failure_1
        ), "The step continue on failure flag was not set correctly."

    def test_step_decorator_display_name(
        self,
        mock_step_entrypoint_pass,
        mock_step_entrypoint_fail,
        mock_step_metadata_name_1,
    ):
        decorated_function_1 = step_decorator(
            name=mock_step_metadata_name_1,
        )(mock_step_entrypoint_pass)

        assert (
            decorated_function_1.metadata.display_name == mock_step_metadata_name_1
        ), "The step's display name not set correctly."

        decorated_function_2 = step_decorator()(mock_step_entrypoint_fail)

        assert (
            decorated_function_2.metadata.display_name
            == mock_step_entrypoint_fail.__name__
        ), "The step's default display name not set correctly."

    def test_step_decorator_get_state(
        self,
        mock_step_entrypoint_pass,
    ):
        decorated_function_1 = step_decorator()(mock_step_entrypoint_pass)

        assert (
            decorated_function_1.state == decorated_function_1.metadata.state
        ), "There's a mismatch between the step's state and the step's metadata state."

    def test_step_log_its_info(
        self, mock_pipeline, step_factory, mock_step_entrypoint_pass, test_logger
    ):
        mock_step = step_factory(
            continue_on_failure=True, entrypoint=mock_step_entrypoint_pass
        )
        with patch.object(test_logger, "info") as mock_info:
            content = "Test logging"
            mock_step.log_step_info(mock_pipeline, test_logger, content)
            expected_log_message = mock_step._format_step_info(
                step_index=mock_step.metadata.index,
                total_number_of_steps=len(mock_pipeline.steps_metadata),
                content=content,
            )
            mock_info.assert_called_with(expected_log_message)

    def test_step_get_logger(self, mock_pipeline, step_factory):
        mock_step = step_factory(continue_on_failure=True, entrypoint=lambda: None)
        mock_pipeline._registered_steps_metadata = [mock_step.metadata]
        mock_pipeline._configure_logging()
        Pipeline.ACTIVE_PIPELINE = mock_pipeline

        step_logger = mock_step._prepare_step_logger(mock_pipeline)
        assert (
            step_logger == mock_pipeline.logger_manager.logger
        ), "The prepared step logger doesn't match the pipeline logger manager's logger."

    def test_step_get_logger_wrong_path(self, mock_pipeline, step_factory):
        mock_step = step_factory(continue_on_failure=True, entrypoint=lambda: None)
        mock_step.metadata.log_file_path = "wrong/path/to/file"
        with pytest.raises(FileNotFoundError):
            _ = mock_step._prepare_step_logger(mock_pipeline)

    def test_step_execution_no_active_pipeline(self, mock_pipeline, step_factory):
        mock_step_success = step_factory(
            continue_on_failure=True, entrypoint=lambda: True
        )
        mock_pipeline._registered_steps_metadata = [mock_step_success.metadata]
        mock_pipeline._configure_logging()

        with pytest.raises(RuntimeError):
            mock_step_success()

    def test_step_execution_success(self, mock_pipeline, step_factory):
        mock_step_success = step_factory(
            continue_on_failure=True, entrypoint=lambda: True
        )
        mock_pipeline._registered_steps_metadata = [mock_step_success.metadata]
        mock_pipeline._configure_logging()
        Pipeline.ACTIVE_PIPELINE = mock_pipeline

        mock_step_success()

        assert mock_step_success._metadata.state == StepState.SUCCESS
        assert mock_pipeline.state == PipelineState.SUCCESS

    def test_step_execution_failed(
        self,
        mock_pipeline,
        step_factory,
        mock_step_entrypoint_pass,
        mock_step_entrypoint_fail,
    ):
        mock_step_failed = step_factory(
            continue_on_failure=True, entrypoint=mock_step_entrypoint_fail
        )
        mock_step_skipped = step_factory(
            continue_on_failure=False, entrypoint=mock_step_entrypoint_pass
        )
        mock_pipeline._registered_steps_metadata = [
            mock_step_failed.metadata,
            mock_step_skipped.metadata,
        ]
        mock_pipeline._configure_logging()
        Pipeline.ACTIVE_PIPELINE = mock_pipeline

        mock_step_failed()
        mock_step_skipped()

        assert mock_step_failed._metadata.state == StepState.FAILED
        assert mock_step_skipped._metadata.state == StepState.SKIPPED
        assert mock_pipeline.state == PipelineState.FAILED

    def test_step_execution_partial_success(
        self,
        mock_pipeline,
        step_factory,
        mock_step_entrypoint_pass,
        mock_step_entrypoint_fail,
    ):
        mock_step_success_1 = step_factory(
            continue_on_failure=True, entrypoint=mock_step_entrypoint_pass
        )
        mock_step_failed = step_factory(
            continue_on_failure=True, entrypoint=mock_step_entrypoint_fail
        )
        mock_step_skipped = step_factory(
            continue_on_failure=False, entrypoint=mock_step_entrypoint_pass
        )

        mock_step_success_2 = step_factory(
            continue_on_failure=True, entrypoint=mock_step_entrypoint_pass
        )
        mock_pipeline._registered_steps_metadata = [
            mock_step_success_1.metadata,
            mock_step_failed.metadata,
            mock_step_skipped.metadata,
            mock_step_success_2.metadata,
        ]
        mock_pipeline._configure_logging()
        Pipeline.ACTIVE_PIPELINE = mock_pipeline

        mock_step_success_1()
        mock_step_failed()
        mock_step_skipped()
        mock_step_success_2()

        assert mock_step_success_1._metadata.state == StepState.SUCCESS
        assert mock_step_failed._metadata.state == StepState.FAILED
        assert mock_step_skipped._metadata.state == StepState.SKIPPED
        assert mock_step_success_2._metadata.state == StepState.SUCCESS

        assert mock_pipeline.state == PipelineState.PARTIAL_SUCCESS
