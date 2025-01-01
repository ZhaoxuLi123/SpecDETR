from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import tifffile
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


def irflyfromfile(img_path, img_num=1,repeat =1) -> np.ndarray:
    """Read an image from bytes.

    Args:
        backend (str | None): The image decoding backend type.
    Returns:
        ndarray: Loaded image array.

    Examples:
    """

    with tifffile.TiffFile(img_path) as tif:
        img = tif.asarray()
        img = np.expand_dims(img, axis=-1)
    img_name = img_path.split('/')[-1]
    try:
        img_id = int(img_name.split('.')[0])
    except:
        img_id = 1
    for i in range(1, img_num):
        img_id_i = img_id - 1
        img_name_i = str(img_id_i).zfill(6)+'.tiff'
        img_path_i = img_path.replace(img_name, img_name_i)
        with tifffile.TiffFile(img_path_i) as tif:
            img_i = tif.asarray()
            img_i = np.expand_dims(img_i, axis=-1)
        img = np.concatenate((img_i, img), axis=-1)
    img = np.repeat(img,repeat,axis=-1)
    return img

@TRANSFORMS.register_module()
class LoadIRFlyImageFromFiles(BaseTransform):
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
        image_num: int = 1,
        repeat: int =1,
        normalized_basis = None,
    ) -> None:
        self.to_float32 = to_float32
        self.image_num = image_num
        self.repeat = repeat
        self.normalized_basis = normalized_basis

    def transform(self, results: dict) -> dict:
        """Transform functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        img = irflyfromfile(results['img_path'],img_num = self.image_num, repeat=self.repeat)
        # up_limit = 3500
        # low_limit = 600
        # new_img = (img - low_limit) / up_limit
        # new_img[new_img > 1] = 1
        # new_img[new_img < 0] = 0
        # img = new_img * 255
        if self.normalized_basis == None:
            img = img/1500
        else:
            img = img/np.array(self.normalized_basis)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, ')
        return repr_str






