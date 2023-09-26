from picsellia import Client
from picsellia.types.enums import InferenceType
from utils import (
    prepare_mask_directories_for_multilabel,
    convert_seperated_multiclass_masks_to_polygons,
)

api_token = ""

organization = ""

host = "https://trial.picsellia.com"

client = Client(api_token=api_token, organization_name=organization, host=host)

dataset = client.get_dataset_version_by_id("")
dataset.set_type(InferenceType.SEGMENTATION)

class_to_pixel_mapping = {}  # Ex: {"car": 1, "plane": 63, "boat": 127}

original_mask_directory = ""
data_directory = ""

prepare_mask_directories_for_multilabel(
    class_to_pixel_mapping=class_to_pixel_mapping,
    mask_directory=original_mask_directory,
)

convert_seperated_multiclass_masks_to_polygons(
    data_directory=data_directory, dataset_version=dataset
)
