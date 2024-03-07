import logging
import os
from typing import Optional, Union, Callable, TypeVar, Any

from poc.enum import PipelineState, StepState
from poc.pipeline_logger import PipelineLogger
from poc.step_metadata import StepMetadata

F = TypeVar("F", bound=Callable[..., None])
logger = logging.getLogger("poc")


class Pipeline:
    ACTIVE_PIPELINE = None

    def __init__(
        self,
        name: str,
        log_folder_path: Union[str, None],
        remove_logs_on_completion: bool,
        entrypoint: F,
    ) -> None:
        self.name = name
        self.pipeline_logger = PipelineLogger(
            pipeline_name=name, log_folder_path=log_folder_path
        )
        self.remove_logs_on_completion = remove_logs_on_completion
        self.entrypoint = entrypoint

        self._is_pipeline_initialized = False
        self._state = PipelineState.PENDING
        self._registered_steps_metadata: [StepMetadata] = []

    @property
    def is_initialized(self) -> bool:
        return self._is_pipeline_initialized

    @property
    def state(self) -> PipelineState:
        """
        The state of the pipeline is determined by the state of its steps.
        """
        if all(
            step_metadata.state == StepState.PENDING
            for step_metadata in self.steps_metadata
        ):
            return PipelineState.PENDING

        elif all(
            step_metadata.state
            in [StepState.RUNNING, StepState.PENDING, StepState.SUCCESS]
            for step_metadata in self.steps_metadata
        ):
            return PipelineState.RUNNING

        elif all(
            step_metadata.state == StepState.SUCCESS
            for step_metadata in self.steps_metadata
        ):
            return PipelineState.SUCCESS

        else:
            for step_metadata in self.steps_metadata:
                if step_metadata.state in [StepState.FAILED, StepState.SKIPPED]:
                    if any(
                        s.state == StepState.SUCCESS
                        for s in self.steps_metadata[
                            self.steps_metadata.index(step_metadata) + 1 :
                        ]
                    ):
                        return PipelineState.PARTIAL_SUCCESS
                    return PipelineState.FAILED

        raise RuntimeError(
            "Pipeline state could not be determined."
            "You need at least one step to determine the pipeline state.",
        )

    @property
    def steps_metadata(self) -> [StepMetadata]:
        return self._registered_steps_metadata

    def __call__(self, *args, **kwargs) -> Any:
        with self:
            _ = self.entrypoint(*args, **kwargs)

            self._is_pipeline_initialized = True
            return self.entrypoint(*args, **kwargs)

    def __enter__(self):
        if Pipeline.ACTIVE_PIPELINE is not None:
            raise RuntimeError(
                "Another pipeline is already active."
                "A pipeline cannot be run within another pipeline."
            )

        Pipeline.ACTIVE_PIPELINE = self

        self.pipeline_logger.configure()
        self._state = PipelineState.RUNNING

    def __exit__(self, exc_type, exc_val, exc_tb):
        Pipeline.ACTIVE_PIPELINE = None

        if self.remove_logs_on_completion:
            self.pipeline_logger.clean()

    def register_step_metadata(
        self,
        step_metadata: StepMetadata,
    ) -> None:
        self._registered_steps_metadata.append(step_metadata)


def pipeline(
    _func: Optional["F"] = None,
    name: Union[str, None] = None,
    log_folder_path: Union[str, None] = None,
    remove_logs_on_completion: bool = True,
) -> Union["Pipeline", Callable[["F"], "Pipeline"]]:
    """Decorator to create a pipeline.

    Args:
        _func: The decorated function.
        name: The name of the pipeline. If left empty, the name of the
            decorated function will be used as a fallback.
        log_folder_path: The path to the log folder. If left empty, a temporary folder will be created.
        remove_logs_on_completion: Whether to remove the logs on completion. Defaults to True.

    Returns:
        A pipeline instance.
    """

    def inner_decorator(func: "F") -> "Pipeline":
        p = Pipeline(
            name=name or func.__name__,
            log_folder_path=log_folder_path,
            remove_logs_on_completion=remove_logs_on_completion,
            entrypoint=func,
        )

        p.__doc__ = func.__doc__
        return p

    return inner_decorator if _func is None else inner_decorator(_func)
