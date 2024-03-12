from poc.step import step


@step
def data_validator(context: dict, dataset_context: dict):
    return dataset_context
