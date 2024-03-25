import functools
import logging
import time
import uuid
from typing import Callable, Union, TypeVar, Optional, Any

from src import Pipeline
from src.enums import StepState, PipelineState
from src.models.steps.step_metadata import StepMetadata

F = TypeVar("F", bound=Callable[..., None])


class Step:
    def __init__(
        self,
        continue_on_failure: bool,
        entrypoint: F,
        metadata: StepMetadata,
    ) -> None:
        """Initialize a step.

        Args:
            continue_on_failure: Tells if the step should be executed, even if the previous steps have failed.
            entrypoint: The function to be executed when the step is called.
            metadata: The metadata associated to the step.
        """
        self.id = metadata.id
        self.step_name = metadata.name
        self.continue_on_failure = continue_on_failure
        self.entrypoint = entrypoint

        self._metadata = metadata

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = None
        current_pipeline = Pipeline.ACTIVE_PIPELINE

        if not current_pipeline:
            raise RuntimeError(
                "No current pipeline running."
                "A step must be run within a function decorated with @pipeline."
            )

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
                    log_content=f"Starting step {self.step_name} ({self.id}):",
                )

                result = self.entrypoint(*args, **kwargs)

            except Exception as e:
                step_logger.error(f"Error in {self.step_name}: {e}", exc_info=True)
                self._metadata.state = StepState.FAILED

            else:
                self._metadata.state = StepState.SUCCESS

            finally:
                execution_time = time.time() - start_time
                self.log_step_info(
                    pipeline=current_pipeline,
                    step_logger=step_logger,
                    log_content=f"{self.step_name} execution time: {execution_time:.3f} seconds",
                )
                self._metadata.execution_time = execution_time

        return None if self.state is StepState.FAILED else result

    @property
    def metadata(self) -> StepMetadata:
        return self._metadata

    @property
    def state(self) -> StepState:
        return self._metadata.state

    def log_step_info(
        self, pipeline: Pipeline, step_logger: logging.Logger, log_content: str
    ) -> None:
        total_number_of_steps = len(pipeline.steps_metadata)
        step_logger.info(
            self._format_step_info(
                step_index=self._metadata.index,
                total_number_of_steps=total_number_of_steps,
                content=log_content,
            )
        )

    def _format_step_info(
        self, step_index: int, total_number_of_steps: int, content: str
    ) -> str:
        return f"({step_index}/{total_number_of_steps}) {content}"

    def _prepare_step_logger(self, pipeline: Pipeline) -> logging.Logger:
        return pipeline.logger_manager.prepare_logger(
            log_file_path=self.metadata.log_file_path
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

    @functools.wraps(_func)
    def inner_decorator(func: "F") -> "Step":
        s_metadata = StepMetadata(
            id=uuid.uuid4(),
            name=func.__name__,
            display_name=name or func.__name__,
            state=StepState.PENDING,
        )
        s = Step(
            continue_on_failure=continue_on_failure,
            entrypoint=func,
            metadata=s_metadata,
        )

        Pipeline.register_step_metadata(step_metadata=s_metadata)
        return s

    return inner_decorator if _func is None else inner_decorator(_func)
