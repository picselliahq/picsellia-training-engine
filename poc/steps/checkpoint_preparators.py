from poc.step import step


@step
def checkpoints_preparator(context: dict, checkpoints_path: str):
    return checkpoints_path
