from poc.pipeline import pipeline
from poc.step import step
from poc.pipelines.demo.test2 import step_3 as step_3b


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
def demo_pipeline():
    val_1 = step_1()
    val_2 = step_2()
    step_3(value_1=val_1, value_2=val_2)
    step_3b(value_1=val_1, value_2=val_2)


if __name__ == "__main__":
    demo_pipeline()
