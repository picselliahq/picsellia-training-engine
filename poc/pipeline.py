import ast
import inspect
from typing import Optional, Union, Callable, TypeVar, Any, List
from poc.enums.state_enums import PipelineState, StepState
from poc.models.logging.logger_manager import LoggerManager
from poc.models.steps.step_metadata import StepMetadata

F = TypeVar("F", bound=Callable[..., None])


class Pipeline:
    ACTIVE_PIPELINE: Optional["Pipeline"] = None
    STEPS_REGISTRY: dict[str, StepMetadata] = {}

    def __init__(
        self,
        name: str,
        log_folder_path: Optional[str],
        remove_logs_on_completion: bool,
        entrypoint: F,
    ) -> None:
        self.name = name
        self.logger_manager = LoggerManager(
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
    def steps_metadata(self) -> List[StepMetadata]:
        return self._registered_steps_metadata

    def __call__(self, *args, **kwargs) -> Any:
        with self:
            self._analyze_and_register_steps()
            self.finalize_initialization()

            return self.entrypoint(*args, **kwargs)

    def __enter__(self):
        if Pipeline.ACTIVE_PIPELINE is not None:
            raise RuntimeError(
                "Another pipeline is already active."
                "A pipeline cannot be run within another pipeline."
            )

        Pipeline.ACTIVE_PIPELINE = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Pipeline.ACTIVE_PIPELINE = None

        if self.remove_logs_on_completion:
            self.logger_manager.clean()

    def register_active_step_metadata(self, step_metadata: StepMetadata) -> None:
        self._registered_steps_metadata.append(step_metadata)

    @staticmethod
    def register_step_metadata(step_metadata: StepMetadata) -> None:
        step_name = step_metadata.name

        if step_name in Pipeline.STEPS_REGISTRY:
            raise ValueError(
                f"More than one step is called '{step_name}'. The step names must be unique."
            )

        Pipeline.STEPS_REGISTRY[step_name] = step_metadata

    def finalize_initialization(self) -> None:
        self.logger_manager.configure(steps_metadata=self.steps_metadata)
        self._is_pipeline_initialized = True
        self._state = PipelineState.RUNNING

    def _analyze_and_register_steps(self) -> None:
        """Analyze the pipeline entrypoint function to identify and register step calls."""
        src = inspect.getsource(self.entrypoint)
        tree = ast.parse(src)

        pipeline_instance = self

        class StepCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    step_func_name = node.func.id
                    if step_func_name in Pipeline.STEPS_REGISTRY:
                        metadata = Pipeline.STEPS_REGISTRY[step_func_name]
                        pipeline_instance.register_active_step_metadata(metadata)

                self.generic_visit(node)

        visitor = StepCallVisitor()
        visitor.visit(tree)


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
