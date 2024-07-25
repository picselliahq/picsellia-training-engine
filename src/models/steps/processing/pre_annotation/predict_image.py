import argparse
import os.path

import cv2
from paddleocr import PaddleOCR


def load_model():
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    return ocr


def print_result(result):
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(f"box: {line[0]}, text: {line[1]}")


def write_result_on_image(image, boxes, txts, scores, result_image_path):
    for i in range(len(boxes)):
        box = boxes[i]
        txt = txts[i]
        score = scores[i]
        cv2.rectangle(
            image,
            (int(box[0][0]), int(box[0][1])),
            (int(box[2][0]), int(box[2][1])),
            (0, 255, 0),
            2,
        )
        combined_text = f"{txt} ({score:.2f})"
        cv2.putText(
            image,
            combined_text,
            (int(box[0][0]), int(box[0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    cv2.imwrite(result_image_path, image)


def get_annotations_from_result(result):
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return boxes, txts, scores


def save_result(result, image_path, result_image_path):
    image = cv2.imread(image_path)
    boxes, txts, scores = get_annotations_from_result(result)
    write_result_on_image(image, boxes, txts, scores, result_image_path)


def predict_image(ocr_model, image_path):
    result = ocr_model.ocr(image_path, cls=True)
    return result


def predict_image_dir(image_dir, result_dir=None):
    ocr_model = load_model()
    for image_filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_filename)
        result = predict_image(ocr_model, image_path)
        print_result(result)
        if result_dir:
            result_image_path = os.path.join(result_dir, image_filename)
            save_result(result, image_path, result_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR inference")
    parser.add_argument("--image_dir", type=str, default="images", help="image dir")
    parser.add_argument("--result_dir", type=str, default="results", help="result dir")
    args = parser.parse_args()

    predict_image_dir(args.image_dir, args.result_dir)
