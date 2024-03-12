import logging
import time

from tqdm import tqdm

from poc.pipeline import pipeline, Pipeline
from poc.step import step


@pipeline(name="pipeline", log_folder_path="poc/logs", remove_logs_on_completion=False)
def training_pipeline(logger: logging.Logger):
    logger.info("Starting pipeline")
    val_1 = step_1()
    val_2 = step_2()
    val_3 = step_3(value_1=val_1, value_2=val_2)
    print(val_3)

    val_4 = step_4(value_1=val_1)
    print(val_4)

    if Pipeline.ACTIVE_PIPELINE is not None:
        print(Pipeline.ACTIVE_PIPELINE.steps_metadata)
        print(f"Pipeline state is: {Pipeline.ACTIVE_PIPELINE.state}")


@step
def step_1() -> float:
    for i in tqdm(range(100)):
        print(i)
    time.sleep(0.5)
    return 0.1


@step(name="step_2_meant_to_fail")
def step_2(logger: logging.Logger) -> float:
    time.sleep(0.7)
    try:
        return 0.9 / 0
    except ZeroDivisionError as e:
        logger.error(f"Error in meant_to_fail(): {e}", exc_info=True)


@step
def step_3(value_1, value_2) -> float:
    print("toto")
    time.sleep(0.8)
    return value_1 + value_2


@step(continue_on_failure=True)
def step_4(value_1) -> float:
    print("shinzo sasageo")
    time.sleep(0.9)
    return value_1 * 2


if __name__ == "__main__":
    training_pipeline()
