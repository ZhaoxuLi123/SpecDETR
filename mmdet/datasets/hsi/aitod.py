# Copyright (c) OpenMMLab. All rights reserved.
# written by lzx

import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset
from mmdet.datasets.coco import CocoDataset



@DATASETS.register_module()
class AITODDataset(CocoDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool', 'vehicle', 'person', 'wind-mill',),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228),
         (0, 60, 100),  (0, 0, 70),  (250, 170, 30),   (100, 170, 30),]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True





