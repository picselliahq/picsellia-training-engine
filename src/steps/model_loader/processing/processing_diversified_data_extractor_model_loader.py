import torch
from open_clip import create_model_and_transforms

from src import step


@step
def diversified_data_extractor_model_loader():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = create_model_and_transforms(
        "ViT-B-16-plus-240", pretrained="laion400m_e32"
    )
    model.to(device)
    return model
