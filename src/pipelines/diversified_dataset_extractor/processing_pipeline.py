from src import pipeline
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.parameters.processing.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_data_extractor,
)
from src.steps.processing.diversified_data_extractor.diversified_data_extractor_processing import (
    diversified_data_extractor_processing,
)


# def fetch_and_prepare_image(url: str) -> Image:
#     """Fetches and prepares an image for processing."""
#     try:
#         with requests.get(url, stream=True) as response:
#             response.raise_for_status()
#             image = Image.open(response.raw)
#             if image.getexif().get(0x0112, 1) != 1:
#                 image = ImageOps.exif_transpose(image)
#             return image.convert("RGB") if image.mode != "RGB" else image
#     except requests.RequestException as e:
#         print(f"Failed to fetch or process image from {url}: {str(e)}")
#         return None
#
#
# def compute_image_tensor(image: Image) -> np.ndarray:
#     """Converts an image to a tensor."""
#     tensor = preprocess(image).unsqueeze(0).to(device)
#     return model.encode_image(tensor).detach().cpu().numpy()
#
#
# def main(api_token, organization_name, host, dist_threshold=5):
#
#     version = dataset.create_version("diversified_datalake")
#     datalake_batch_size = 1000
#     datalake_current_offset = 0
#
#     datalake = client.get_datalake()
#     datalake_size = datalake.sync()["size"]
#
#     phash_dict = {}
#     tensor_list = []
#     kd_tree = None
#
#     skipped_asset_number = 0
#
#     while datalake_current_offset < datalake_size:
#         print(
#             f"Starting datalake batch {datalake_current_offset/datalake_batch_size + 1}/{math.ceil(datalake_size/datalake_batch_size)}"
#         )
#         try:
#             data_list = datalake.list_data(
#                 limit=datalake_batch_size, offset=datalake_current_offset
#             )
#         except Exception:
#             data_list = datalake.list_data(
#                 limit=datalake_batch_size, offset=datalake_current_offset
#             )
#
#         batch_to_upload = []
#
#         for data in tqdm(data_list, desc="Processing Images"):
#             try:
#                 image = fetch_and_prepare_image(data.url)
#
#                 if image:
#                     img_hash = str(phash(image))
#
#                     if img_hash not in phash_dict:
#                         phash_dict[img_hash] = True
#                         tensor = compute_image_tensor(image)
#                         if kd_tree is None:
#                             kd_tree = cKDTree(tensor)
#                             tensor_list.append(tensor)
#                             batch_to_upload.append(data)
#                         else:
#                             dist, _ = kd_tree.query(tensor)
#
#                             if dist > dist_threshold:
#                                 kd_tree = cKDTree(np.vstack([kd_tree.data, tensor]))
#                                 tensor_list.append(tensor)
#                                 batch_to_upload.append(data)
#                             else:
#                                 skipped_asset_number += 1
#                     else:
#                         skipped_asset_number += 1
#                 else:
#                     skipped_asset_number += 1
#
#             except Exception as e:
#                 skipped_asset_number += 1
#                 print(f"Skipped asset due to exception: {e}")
#
#         version.add_data(batch_to_upload)
#         datalake_current_offset += datalake_batch_size
#         print(
#             f"Uploaded {datalake_current_offset - skipped_asset_number} | Skipped {skipped_asset_number}"
#         )
#
#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Upload dataset to Picsellia")
#     parser.add_argument(
#         "--api_token", required=True, help="API token of the Picsellia account"
#     )
#     parser.add_argument(
#         "--dist_threshold", required=True, help="API token of the Picsellia account"
#     )
#     parser.add_argument("--organization_name", required=True, help="Organization name")
#     parser.add_argument(
#         "--host",
#         default="https://app.picsellia.com",
#         help="Host of the Picsellia account",
#     )
#     args = parser.parse_args()
#     main(args.api_token, args.organization_name, args.host)


def get_context() -> (
    PicselliaProcessingContext[ProcessingDiversifiedDataExtractorParameters]
):
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingDiversifiedDataExtractorParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def diversified_data_extractor_pipeline() -> None:
    dataset_context = processing_data_extractor(skip_asset_listing=True)
    diversified_data_extractor_processing(dataset_context=dataset_context)


if __name__ == "__main__":
    diversified_data_extractor_pipeline()
