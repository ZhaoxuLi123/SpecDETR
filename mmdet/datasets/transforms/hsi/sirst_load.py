import random
from typing import Optional, Tuple, Union

import torch
import numpy as np
import mmcv
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes
import mmengine.fileio as fileio
import cv2
import matplotlib.pyplot as plt

# def hsifromfile(img_path, backend='npy' ) -> np.ndarray:
#     """Read an image from bytes.
#
#     Args:
#         backend (str | None): The image decoding backend type.
#     Returns:
#         ndarray: Loaded image array.
#
#     Examples:
#     """
#     if backend =='npy':
#         img = np.load(img_path)
#         return img

@TRANSFORMS.register_module()
class LoadSIRSTImageFromFiles(BaseTransform):
    """Load multi-channel images from a list of separate channel files.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(
        self,
        to_float32: bool = False,
        normalized_basis = None,
        range_change = False,
    ) -> None:
        self.to_float32 = to_float32
        self.normalized_basis = normalized_basis
        self.range_change = range_change
    def transform(self, results: dict) -> dict:
        """Transform functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        # img = hsifromfile(results['img_path'])
        img = plt.imread(results['img_path'])
        # up_limit = 3500
        # low_limit = 600
        # new_img = (img - low_limit) / up_limit
        # new_img[new_img > 1] = 1
        # new_img[new_img < 0] = 0
        # img = new_img * 255
        # if self.normalized_basis == None:
        #     img = img/500
        # else:
        # img = img/255
        # img = img[:,:,0]
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        # print(img.shape)
        if img.shape[-1] !=3:
            img = img[:,:,:3]
        if self.to_float32:
            img = img.astype(np.float32)
        img = (img-np.min(img))/(np.max(img)-np.min(img)+1e-8)
        if self.range_change ==True:
            img = (0.7+random.random()*0.3)*img
            # img = img + random.random()*0.2
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, ')
        return repr_str





