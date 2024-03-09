import os

from picsellia import Artifact

from poc.step import step


@step
def checkpoints_extractor(context: dict):
    model_file: Artifact = context["experiment"].get_artifact("weights")
    model_file.download(
        target_path=os.path.join(context["experiment"].name, "checkpoints")
    )
    return os.path.abspath(
        os.path.join(context["experiment"].name, "checkpoints", model_file.filename)
    )
