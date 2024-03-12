import logging
import time
import uuid
from typing import Callable, Union, TypeVar, Optional, Any

from poc.enums.state_enums import StepState, PipelineState
from poc.pipeline import Pipeline
from poc.models.steps.step_metadata import StepMetadata

F = TypeVar("F", bound=Callable[..., None])


class Step:
    def __init__(
        self,
        name: str,
        continue_on_failure: bool,
        entrypoint: F,
    ) -> None:
        """Initializes a step.

        Args:
            name: The name of the step.
            entrypoint: The entrypoint function of the pipeline.
        """
        self.id = uuid.uuid4()
        self.step_name = name
        self.continue_on_failure = continue_on_failure
        self.entrypoint = entrypoint

        self._metadata = self.initialize_metadata()

    @property
    def state(self) -> StepState:
        return self._metadata.state

    @property
    def metadata(self) -> StepMetadata:
        return self._metadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = None
        current_pipeline = Pipeline.ACTIVE_PIPELINE

        if not current_pipeline:
            raise RuntimeError(
                "No current pipeline running."
                "A step must be run within a function decorated with @pipeline."
            )

        if not current_pipeline.is_initialized:
            current_pipeline.register_step_metadata(step_metadata=self._metadata)
            return self._metadata

        else:
            self._metadata.state = StepState.RUNNING
            step_logger = self._prepare_step_logger(pipeline=current_pipeline)

        if (
            current_pipeline.state
            not in [PipelineState.RUNNING, PipelineState.PARTIAL_SUCCESS]
            and not self.continue_on_failure
        ):
            self._metadata.state = StepState.SKIPPED
            self.log_step_info(
                pipeline=current_pipeline,
                step_logger=step_logger,
                log_content=f"{self.step_name} was skipped.",
            )

        else:
            start_time = time.time()

            try:
                self.log_step_info(
                    pipeline=current_pipeline,
                    step_logger=step_logger,
                    log_content=f"Starting step {self.step_name} ({self.id})",
                )

                result = self.entrypoint(*args, **kwargs)

            except Exception as e:
                step_logger.error(f"Error in {self.step_name}: {e}", exc_info=True)
                self._metadata.state = StepState.FAILED

            else:
                self._metadata.state = StepState.SUCCESS

            finally:
                execution_time = time.time() - start_time
                step_logger.info(
                    f"{self.step_name} execution time: {execution_time:.3f} seconds"
                )
                self._metadata.execution_time = execution_time

        return None if self.state is StepState.FAILED else result

    def _prepare_step_logger(self, pipeline: Pipeline) -> logging.Logger:
        return pipeline.logger_manager.prepare_step_logger(
            log_file_path=self.metadata.log_file_path
        )

    def log_step_info(
        self, pipeline: Pipeline, step_logger: logging.Logger, log_content: str
    ) -> None:
        total_number_of_steps = len(pipeline.steps_metadata)
        step_logger.info(
            f"({self.metadata.index}/{total_number_of_steps}) {log_content}:"
        )

    def initialize_metadata(self) -> StepMetadata:
        return StepMetadata(
            id=self.id,
            name=self.step_name,
            state=StepState.PENDING,
            execution_time=0,
        )


def step(
    _func: Optional["F"] = None,
    name: Union[str, None] = None,
    continue_on_failure: bool = False,
) -> Union["Step", Callable[["F"], "Step"]]:
    """Decorator to create a step.

    Args:
        _func: The decorated function.
        name: The name of the step. If left empty, the name of the
            decorated function will be used as a fallback.
        continue_on_failure: Tells if the step should be executed, even if the previous steps have failed

    Returns:
        A pipeline instance.
    """

    def inner_decorator(func: "F") -> "Step":
        s = Step(
            name=name or func.__name__,
            continue_on_failure=continue_on_failure,
            entrypoint=func,
        )

        s.__doc__ = func.__doc__
        return s

    return inner_decorator if _func is None else inner_decorator(_func)
