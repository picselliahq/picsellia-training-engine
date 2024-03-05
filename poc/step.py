import logging
import time
from typing import Callable, Union, TypeVar, Optional, Any

from poc.enum import StepState, PipelineState
from poc.pipeline import Pipeline

F = TypeVar("F", bound=Callable[..., None])
logger = logging.getLogger("poc")


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
        self.step_name = name
        self.continue_on_failure = continue_on_failure
        self.entrypoint = entrypoint

        self._state = StepState.RUNNING

    @property
    def state(self) -> StepState:
        return self._state

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = None
        current_pipeline = Pipeline.ACTIVE_PIPELINE

        if not current_pipeline:
            raise RuntimeError(
                "No current pipeline running."
                "A step must be run within a function decorated with @pipeline."
            )

        if (
            current_pipeline.state
            not in [PipelineState.RUNNING, PipelineState.PARTIAL_SUCCESS]
            and not self.continue_on_failure
        ):
            self._state = StepState.SKIPPED
            logger.warning(f"{self.step_name} was skipped.")

        else:
            start_time = time.time()

            try:
                result = self.entrypoint(*args, **kwargs)

            except Exception as e:
                logger.error(f"Error in {self.step_name}: {e}", exc_info=True)
                self._state = StepState.FAILED

            else:
                self._state = StepState.SUCCESS
                execution_time = time.time() - start_time
                logger.info(
                    f"{self.step_name} execution time: {execution_time:.3f} seconds"
                )

        current_pipeline.register_step_execution(self.step_name, self.state)
        return None if self.state is StepState.FAILED else result


def step(
    _func: Optional["F"] = None,
    name: str | None = None,
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
