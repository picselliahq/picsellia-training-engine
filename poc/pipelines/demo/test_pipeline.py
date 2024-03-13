from poc.pipeline import pipeline
from poc.step import step


@step
def step_1() -> float:
    return 0.1


@step
def step_2() -> int:
    return 1


@step
def step_3(value_1, value_2) -> float:
    return value_1 + value_2


@pipeline
def demo_pipeline(
    context_preparator=step_1, model_loader=step_2, model_exporter=step_3
):
    val_1 = step_1()
    val_2 = step_2()
    step_3(value_1=val_1, value_2=val_2)
