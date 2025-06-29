import copy
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
from PIL import Image

def irairfromfilev1(img_path, img_num=1,repeat =1) -> np.ndarray:
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
        img_id_i = img_id - i
        img_name_i = str(img_id_i).zfill(6)+'.tiff'
        img_path_i = img_path.replace(img_name, img_name_i)
        with tifffile.TiffFile(img_path_i) as tif:
            img_i = tif.asarray()
            img_i = np.expand_dims(img_i, axis=-1)
        img = np.concatenate((img,  img_i), axis=-1)
    img = np.repeat(img, repeat, axis=-1)
    return img


def irairfromfile(img_path,
                  frame_num=1,
                  current_frame_repeat=1,
                  temporal_filter=False,
                  spatial_temporal_concat=False,
                  normalized_basis=2000
                  ) -> np.ndarray:
    """Read an image from bytes.

    Args:
        backend (str | None): The image decoding backend type.
    Returns:
        ndarray: Loaded image array.

    Examples:
    """
    img_format = img_path.split('.')[-1]
    if img_format == 'tiff':
        with tifffile.TiffFile(img_path) as tif:
            img = tif.asarray()
    else:
        img = Image.open(img_path)
        img = np.array(img)
    img = np.expand_dims(img, axis=-1)
    img_c = copy.deepcopy(img)
    img_name = img_path.split('/')[-1]
    try:
        img_id = int(img_name.split('.')[0])
    except:
        img_id = 1
    assert img_id > frame_num, 'img_id shoudl be bigger than frame_num'

    img = np.repeat(img, current_frame_repeat, axis=-1)
    for i in range(1, frame_num+1):
        img_id_i = img_id - i
        img_name_i = str(img_id_i).zfill(6)+'.'+img_format
        img_path_i = img_path.replace(img_name, img_name_i)
        if img_format == 'tiff':
            with tifffile.TiffFile(img_path_i) as tif:
                img_i = tif.asarray()
        else:
            img_i = Image.open(img_path_i)
            img_i = np.array(img_i)
        img_i = np.expand_dims(img_i, axis=-1)
        img = np.concatenate((img, img_i), axis=-1)
    img = img / np.array(normalized_basis)
    if temporal_filter:
        img = img*np.array(normalized_basis)
        median_values = np.median(img[:, :, 1:], axis=2)
        tf_result = img[:, :, 0] - median_values
        tf_result = np.expand_dims(tf_result, axis=-1)
        if spatial_temporal_concat:
            img = np.concatenate((tf_result, img_c/ np.array(normalized_basis)), axis=-1)
        else:
            img = tf_result
        # from matplotlib.pylab import mpl
        # import matplotlib.pyplot as plt
        # mpl.use('Qt5Agg')
        # plt.figure()
        # plt.imshow(tf_result, cmap='gray')
        # plt.show()

    return img


@TRANSFORMS.register_module()
class LoadIRAirImageFromFiles(BaseTransform):
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
        frame_num: int = 0,
        current_frame_repeat: int = 1,
        temporal_filter: bool = False,
        spatial_temporal_concat: bool = False,
        normalized_basis=None,
    ) -> None:
        self.to_float32 = to_float32
        self.frame_num = frame_num
        self.current_frame_repeat = current_frame_repeat
        self.temporal_filter = temporal_filter
        assert (not self.temporal_filter) or (self.temporal_filter and self.current_frame_repeat == 1),\
            'There is a conflict between current_frame_repeat and temporal_filter '
        self.normalized_basis = normalized_basis
        self.spatial_temporal_concat = spatial_temporal_concat

    def transform(self, results: dict) -> dict:
        """Transform functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """
        if self.normalized_basis is None:
            normalized_basis = 2000
        else:
            normalized_basis = self.normalized_basis
        img = irairfromfile(results['img_path'], frame_num=self.frame_num,
                            current_frame_repeat=self.current_frame_repeat,
                            temporal_filter=self.temporal_filter,
                            spatial_temporal_concat=self.spatial_temporal_concat,
                            normalized_basis=normalized_basis)
        # up_limit = 3500
        # low_limit = 600
        # new_img = (img - low_limit) / up_limit
        # new_img[new_img > 1] = 1
        # new_img[new_img < 0] = 0
        # img = new_img * 255
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






