from poc.step import step


@step
def weights_preparator(context: dict, weights_path: str):
    return weights_path
