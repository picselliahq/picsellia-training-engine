import requests
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from open_clip import create_model_and_transforms
from picsellia import Client
from imagehash import phash
from scipy.spatial import cKDTree

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(
    "ViT-B-16-plus-240", pretrained="laion400m_e32"
)
model.to(device)


def fetch_and_prepare_image(url: str) -> Image:
    """Fetches and prepares an image for processing."""
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            image = Image.open(response.raw)
            if image.getexif().get(0x0112, 1) != 1:
                image = ImageOps.exif_transpose(image)
            return image.convert("RGB") if image.mode != "RGB" else image
    except requests.RequestException as e:
        print(f"Failed to fetch or process image from {url}: {str(e)}")
        return None


def compute_image_tensor(image: Image) -> np.ndarray:
    """Converts an image to a tensor."""
    tensor = preprocess(image).unsqueeze(0).to(device)
    return model.encode_image(tensor).detach().cpu().numpy()


def main(api_token, organization_name, host):
    client = Client(api_token=api_token, organization_name=organization_name, host=host)
    dataset_name = "test_dataset"
    client.get_dataset(dataset_name).delete()
    dataset = client.create_dataset(dataset_name)
    version = dataset.create_version("initial")
    datalake = client.get_datalake(name="RTE")
    data_list = datalake.list_data()

    phash_dict = {}
    tensor_list = []
    kd_tree = None
    data_to_upload = []

    for data in tqdm(data_list, desc="Processing Images"):
        image = fetch_and_prepare_image(data.url)
        if image:
            img_hash = str(phash(image))
            if img_hash not in phash_dict:
                phash_dict[img_hash] = True
                tensor = compute_image_tensor(image)
                if kd_tree is None:
                    kd_tree = cKDTree(tensor)
                    tensor_list.append(tensor)
                    data_to_upload.append(data)
                else:
                    dist, _ = kd_tree.query(tensor)
                    print(dist)
                    if dist > 5:  # Example threshold for "new" content
                        kd_tree = cKDTree(np.vstack([kd_tree.data, tensor]))
                        tensor_list.append(tensor)
                        data_to_upload.append(data)

    if data_to_upload:
        version.add_data(data_to_upload)
        print(f"Uploaded {len(data_to_upload)} new items to dataset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload dataset to Picsellia")
    parser.add_argument(
        "--api_token", required=True, help="API token of the Picsellia account"
    )
    parser.add_argument("--organization_name", required=True, help="Organization name")
    parser.add_argument(
        "--host",
        default="https://staging.picsellia.com",
        help="Host of the Picsellia account",
    )
    args = parser.parse_args()
    main(args.api_token, args.organization_name, args.host)
