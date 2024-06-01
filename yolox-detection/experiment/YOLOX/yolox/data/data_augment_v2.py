import cv2
import imgaug.augmenters as iaa
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from YOLOX.yolox.utils import xyxy2cxcywh
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class TrainTransformV2:
    def __init__(self, max_labels=50):
        self.max_labels = max_labels
        self.aug = self._load_custom_augmentations()

    def _load_custom_augmentations(self):
        """Create a sequence of imgaug augmentations"""

        def sometimes(aug):
            return iaa.Sometimes(0.5, aug)

        return iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(
                    iaa.CropAndPad(
                        percent=(-0.05, 0.1), pad_mode="constant", pad_cval=0
                    )
                ),
                sometimes(
                    iaa.Affine(
                        # scale images to 80-120% of their size, individually per axis
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # translate by -20 to +20 percent (per axis)
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        # use nearest neighbour or bilinear interpolation (fast)
                        order=[0, 1],
                        # if mode is constant, use a cval between 0 and 255
                        cval=0,
                        # use any of scikit-image's warping modes
                        # (see 2nd image from the top for examples)
                        mode="constant",
                    )
                ),
                # execute 0 to 5 of the following (less important) augmenters per
                # image don't execute all of them, as that would often be way too
                # strong
                iaa.SomeOf(
                    (0, 5),
                    [
                        # convert images into their superpixel representation
                        iaa.OneOf(
                            [
                                # blur images with a sigma between 0 and 3.0
                                iaa.GaussianBlur((0, 2.0)),
                                # blur image using local means with kernel sizes
                                # between 2 and 7
                                iaa.AverageBlur(k=(2, 5)),
                                # blur image using local medians with kernel sizes
                                # between 2 and 7
                                iaa.MedianBlur(k=(1, 3)),
                            ]
                        ),
                        iaa.Sharpen(
                            alpha=(0, 0.5), lightness=(0.9, 1.2)
                        ),  # sharpen images
                        # add gaussian noise to images
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                        # change brightness of images (by -10 to 10 of original value)
                        iaa.Add((-10, 10), per_channel=0.5),
                        # sometimes move parts of the image around
                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.04))),
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                        iaa.ElasticTransformation(alpha=(10, 200), sigma=15.0),
                    ],
                    random_order=True,
                ),
            ],
            random_order=True,
        )

    def preproc(self, img, input_size, swap=(2, 0, 1)):
        """Resize and normalize the image as per YOLOX standards."""
        if len(img.shape) == 3:
            padded_img = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, _ = self.preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)

        bbs = BoundingBoxesOnImage(
            [BoundingBox(*box) for box in boxes], shape=image.shape
        )
        image_aug, bbs_aug = self.aug(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.clip_out_of_image()

        valid_indices = [
            i
            for i, bbox in enumerate(bbs_aug.bounding_boxes)
            if bbox.is_fully_within_image(image)
        ]
        labels = labels[valid_indices]
        bbs_aug = bbs_aug.remove_out_of_image()

        boxes = np.array(
            [[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bbs_aug.bounding_boxes]
        )

        height, width, _ = image_aug.shape
        image_t, r_ = self.preproc(image_aug, input_dim)

        # All the bbox have disappeared due to a data augmentation
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            return image_t, targets

        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = self.preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels

    def visualize_image_with_boxes(self, image, boxes):
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for box in boxes:
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()
