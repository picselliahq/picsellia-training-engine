from poc.step import step


@step
def model_register(context: dict, weights_name: str, weights_path: str):
    artifact = context["experiment"].store(weights_name, weights_path)
    return artifact
