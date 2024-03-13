import os

from picsellia import Artifact

from poc.step import step


@step(name="Extract the weights", continue_on_failure=True)
def weights_extractor(context: dict) -> str:
    model_file: Artifact = context["experiment"].get_artifact("weights")
    model_file.download(target_path=os.path.join(context["experiment"].name, "weights"))
    return os.path.abspath(
        os.path.join(context["experiment"].name, "weights", model_file.filename)
    )
