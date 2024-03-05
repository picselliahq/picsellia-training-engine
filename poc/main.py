import logging
import time

from poc.log_handler import CustomLogHandler
from poc.pipeline import pipeline, Pipeline
from poc.step import step


@pipeline
def training_pipeline():
    val_1 = step_1()
    val_2 = step_2()
    val_3 = step_3(value_1=val_1, value_2=val_2)
    print(val_3)

    val_4 = step_4(value_1=val_1)
    print(val_4)

    if Pipeline.ACTIVE_PIPELINE is not None:
        print(Pipeline.ACTIVE_PIPELINE.step_statuses)
        print(Pipeline.ACTIVE_PIPELINE.state)


@step
def step_1() -> float:
    time.sleep(0.5)
    return 0.1


@step(name="meant_to_fail()")
def step_2() -> float:
    time.sleep(0.7)
    return 0.9 / 0


@step
def step_3(value_1, value_2) -> float:
    time.sleep(0.8)
    return value_1 + value_2


@step
def step_4(value_1) -> float:
    time.sleep(0.9)
    return value_1 * 2


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[CustomLogHandler()]
    )
    training_pipeline()
