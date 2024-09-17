import argparse
import logging

import yaml

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat
from minigpt4.models.minigpt_v2 import MiniGPTv2

logger = logging.getLogger(__name__)


def minigpt_load_model(
    weights_path_to_load: str, config_path_to_load: str, device: str
):
    with open(config_path_to_load, "r") as f:
        config = yaml.safe_load(f)

    print(f"model architecture: {config['model_architecture']}")

    model_cls = registry.get_model_class(config["model_architecture"])

    if model_cls is None:
        print(f"mapping of registry: {registry.mapping}")
        model_cls = MiniGPTv2()

    if config["model_architecture"] == "minigpt4":
        "src/pipelines/paddle_ocr/PaddleOCR/tools/train.py"
        eval_config_path = "src/pipelines/datalake_autotagging/MiniGPT4/eval_configs/minigpt4_eval.yaml"
    elif config["model_architecture"] == "minigpt4_llama2":
        eval_config_path = "src/pipelines/datalake_autotagging/MiniGPT4/eval_configs/minigpt4_llama2_eval.yaml"
    elif config["model_architecture"] == "minigpt_v2":
        eval_config_path = "src/pipelines/datalake_autotagging/MiniGPT4/eval_configs/minigptv2_eval.yaml"
    else:
        raise ValueError("Model version not supported")

    args = argparse.Namespace()
    args.cfg_path = eval_config_path
    args.gpu_id = 0
    args.options = []

    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.arch = config["model_architecture"]
    model_config.image_size = config["image_size"]
    model_config.llama_model = config["llama_model"]
    model_config.num_query_token = config["num_query_token"]
    model_config.ckpt = weights_path_to_load
    model_config.low_resource = False

    model = model_cls.from_config(model_config).to(device)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )

    model = model.eval()

    chat = Chat(model=model, vis_processor=vis_processor, device=device)

    return chat
