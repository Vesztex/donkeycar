import albumentations.core.transforms_interface
import cv2
import numpy as np
import logging
import albumentations as A
from albumentations import GaussianBlur
from albumentations.augmentations.crops.transforms import CropAndPad
from albumentations.augmentations.transforms import RandomBrightnessContrast, \
    Lambda

from donkeycar.config import Config


logger = logging.getLogger(__name__)


class Augmentations(object):
    """
    Some ready to use image augumentations.
    """

    @classmethod
    def crop(cls, left, right, top, bottom, keep_size=True):
        """
        The image augumentation sequence.
        Crops based on a region of interest among other things.
        left, right, top & bottom are the number of pixels to crop.
        """
        augmentation = CropAndPad(px=(-top, -right, -bottom, -left),
                                  keep_size=keep_size)
        return augmentation

    @classmethod
    def trapezoidal_mask(cls, lower_left, lower_right, upper_left,
                         upper_right, min_y, max_y):
        """
        Uses a binary mask to generate a trapezoidal region of interest.
        Especially useful in filtering out uninteresting features from an
        input image.
        """
        def _transform_image(image, **kwargs):
            mask = np.zeros(image.shape, dtype=np.int32)
            # # # # # # # # # # # # #
            #       ul     ur          min_y
            #
            #
            #
            #    ll             lr     max_y
            points = [
                [upper_left, min_y],
                [upper_right, min_y],
                [lower_right, max_y],
                [lower_left, max_y]
            ]
            cv2.fillConvexPoly(mask,
                               np.array(points, dtype=np.int32),
                               [255, 255, 255])
            mask = np.asarray(mask, dtype='bool')
            masked = np.multiply(image, mask)
            return masked

        augmentation = Lambda(image=_transform_image)
        return augmentation


class ImageAugmentation:
    def __init__(self, cfg, key, prob=0.5, always_apply=False):
        aug_list = getattr(cfg, key, [])
        augmentations = [ImageAugmentation.create(a, cfg, prob, always_apply)
                         for a in aug_list]
        self.augmentations = A.Compose(augmentations)

    @classmethod
    def create(cls, aug_type: str, config: Config, prob, always) -> \
            albumentations.core.transforms_interface.BasicTransform:
        """ Augmenatition factory. Cropping and trapezoidal mask are
            transfomations which should be applied in training, validation
            and inference. Multiply, Blur and similar are augmentations
            which should be used only in training. """

        if aug_type == 'CROP':
            logger.info(f'Creating augmentation {aug_type} with ROI_CROP '
                        f'L: {config.ROI_CROP_LEFT}, '
                        f'R: {config.ROI_CROP_RIGHT}, '
                        f'B: {config.ROI_CROP_BOTTOM}, '
                        f'T: {config.ROI_CROP_TOP}')

            return Augmentations.crop(left=config.ROI_CROP_LEFT,
                                      right=config.ROI_CROP_RIGHT,
                                      bottom=config.ROI_CROP_BOTTOM,
                                      top=config.ROI_CROP_TOP,
                                      keep_size=True)
        elif aug_type == 'TRAPEZE':
            logger.info(f'Creating augmentation {aug_type}')
            return Augmentations.trapezoidal_mask(
                        lower_left=config.ROI_TRAPEZE_LL,
                        lower_right=config.ROI_TRAPEZE_LR,
                        upper_left=config.ROI_TRAPEZE_UL,
                        upper_right=config.ROI_TRAPEZE_UR,
                        min_y=config.ROI_TRAPEZE_MIN_Y,
                        max_y=config.ROI_TRAPEZE_MAX_Y)

        elif aug_type == 'BRIGHTNESS':
            b_limit = getattr(config, 'AUG_BRIGHTNESS_RANGE', 0.2)
            logger.info(f'Creating augmentation {aug_type} {b_limit}')
            return RandomBrightnessContrast(brightness_limit=b_limit,
                                            contrast_limit=b_limit,
                                            p=prob, always_apply=always)

        elif aug_type == 'BLUR':
            b_range = getattr(config, 'AUG_BLUR_RANGE', 3)
            logger.info(f'Creating augmentation {aug_type} {b_range}')
            return GaussianBlur(sigma_limit=b_range, blur_limit=(13, 13),
                                p=prob, always_apply=always)

    # Parts interface
    def run(self, img_arr):
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr

