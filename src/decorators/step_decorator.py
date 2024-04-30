import logging
import time
import uuid
from typing import Callable, Union, TypeVar, Optional, Any, overload

from src import Pipeline
from src.enums import StepState, PipelineState
from src.models.steps.step_metadata import StepMetadata

F = TypeVar("F", bound=Callable[..., Any])


class Step:
    def __init__(
        self,
        continue_on_failure: bool,
        entrypoint: F,
        metadata: StepMetadata,
    ) -> None:
        """Initializes a step.

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
        """Handles a step's call.

        This method first checks if the step can be executed.
        If yes, it runs the step's entrypoint function, otherwise it skips the step.

        Args:
            *args: Entrypoint function arguments.
            **kwargs: Entrypoint function keyword arguments.

        Returns:
            The outputs of the entrypoint function call.

        Raises:
            RuntimeError: If no current pipeline is running.
        """
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
        """The metadata associated to the step.

        Returns:
            The metadata associated to the step.
        """
        return self._metadata

    @property
    def state(self) -> StepState:
        """The step's current state.

        Returns:
            The step's current state.
        """
        return self._metadata.state

    def log_step_info(
        self, pipeline: Pipeline, step_logger: logging.Logger, log_content: str
    ) -> None:
        """Wrapper to log step information.

        Args:
            pipeline: The pipeline the step belongs to.
            step_logger: The logger to use for logging.
            log_content: The content to log.
        """
        total_number_of_steps = len(pipeline.steps_metadata)
        step_logger.info(
            self._format_step_info(
                step_index=self._metadata.index,
                total_number_of_steps=total_number_of_steps,
                content=log_content,
            )
        )

    def _format_step_info(
        self, step_index: Optional[int], total_number_of_steps: int, content: str
    ) -> str:
        """Formats the step information to be logged.

        Formats the step information to be logged in the following format: (step_index/total_number_of_steps) content.
        If the step index is not provided, it will be replaced by a question mark.

        Args:
            step_index: The step's index, ranging from 1 to the total number of steps.
            total_number_of_steps: The total number of steps in the pipeline.
            content: The content to log.

        Returns:
            The formatted step information.
        """

        return (
            f"({step_index if step_index else '?'}/{total_number_of_steps}) {content}"
        )

    def _prepare_step_logger(self, pipeline: Pipeline) -> logging.Logger:
        """Prepares the logger for the step by configuring the available logger's handlers.

        Args:
            pipeline: The pipeline the step belongs to.

        Returns:
            A ready to be used logger.
        """
        return pipeline.logger_manager.prepare_logger(
            log_file_path=self.metadata.log_file_path
        )


@overload
def step(_func: F) -> Step:  # pragma: no cover
    ...


@overload
def step(
    *, name: Optional[str] = None, continue_on_failure: bool = False
) -> Callable[[F], Step]:  # pragma: no cover
    ...


def step(
    _func: Optional["F"] = None,
    name: Optional[str] = None,
    continue_on_failure: bool = False,
) -> Union["Step", Callable[["F"], "Step"]]:
    """Decorator to create a step.
    The step will automatically be registered in the current pipeline and log its content inside a dedicated log file.

    Args:
        _func: The decorated function.
        name: The name of the step. If left empty, the name of the
            decorated function will be used as a fallback.
        continue_on_failure: Tells if the step should be executed, even if the previous steps have failed

    Returns:
        A pipeline instance.
    """

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
        s.__doc__ = func.__doc__

        Pipeline.register_step_metadata(step_metadata=s_metadata)
        return s

    return inner_decorator if _func is None else inner_decorator(_func)
