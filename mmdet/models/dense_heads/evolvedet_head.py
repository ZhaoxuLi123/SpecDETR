# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path
from typing import Dict, List, Tuple, Optional
from torch import Tensor
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, bbox_overlaps
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..utils import multi_apply
from ..layers import inverse_sigmoid
from .detr_head import DETRHead
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,OptConfigType,
                         OptMultiConfig, reduce_mean)
from mmcv.ops import nms, batched_nms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
import scipy.io as sio
import os
from ..losses import QualityFocalLoss
# def adjust_bbox_to_pixel(bboxes: Tensor):
#     # 向下取整得到目标的左上角坐标
#     adjusted_bboxes = torch.floor(bboxes)
#     # 向上取整得到目标的右下角坐标
#     adjusted_bboxes[:, 2:] = torch.ceil(bboxes[:, 2:])
#     return adjusted_bboxes


def adjust_bbox_to_pixel(bboxes: Tensor):
    # 四舍五入取整坐标
    adjusted_bboxes = torch.round(bboxes)
    return adjusted_bboxes

@MODELS.register_module()
class EvloveDetHead(BaseModule):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int = 256,
                 decoder_embed_dims: int = 256,
                 num_reg_fcs: int = 2,
                 center_feat_indice: int=1,
                 sync_cls_avg_factor: bool = False,
                 use_nms: bool = False,
                 score_threshold: float = 0.0,
                 class_wise_nms: bool = True,
                 test_nms: OptConfigType = dict(type='nms', iou_threshold=0.01, ),
                 loss_cls: ConfigType = dict(
                                    type='FocalLoss',
                                    use_sigmoid=True,
                                    gamma=2.0,
                                    alpha=0.25,
                                    loss_weight=1.0),
                 loss_center_cls: ConfigType = dict(
                                    type='FocalLoss',
                                    use_sigmoid=True,
                                    gamma=2.0,
                                    alpha=0.25,
                                    loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
                 loss_iou: OptConfigType = None,
                 loss_seg: ConfigType = dict(
                                    type='FocalLoss',
                                    use_sigmoid=True,
                                    gamma=2.0,
                                    alpha=0.25,
                                    loss_weight=1.0),
                 # loss_seg: ConfigType = dict(type='L1Loss', loss_weight=1.0),
                 loss_abu: ConfigType = dict(type='L1Loss', loss_weight=1.0),
                 train_cfg: ConfigType = dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         match_costs=[
                             dict(type='ClassificationCost', weight=1.),
                             dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ])),
                 test_cfg: ConfigType = dict(max_per_img=100),
                 init_cfg: OptMultiConfig = None,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 as_two_stage: bool = False,
                 pre_bboxes_round: bool = True,
                 neg_hard_num: int = 0,
                 seg_neg_hard_num: int = 0,
                 loss_center_th: float = 0.2,
                 loss_iou_th: float = 0.3,
                 center_ds_ratio: int = 1,
                 use_center: bool = True,
                 predict_segmentation: bool = False,
                 predict_abundance: bool = False,
                 save_path: Optional[str]= None,
                 mask_threshold:float = 0.5,
                 mask_extend_pixel: int = 2,
                 ) -> None:
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        self.pre_bboxes_round = pre_bboxes_round
        self.score_threshold = score_threshold
        self.loss_center_th = loss_center_th
        self.loss_iou_th = loss_iou_th
        self.center_feat_indice = center_feat_indice
        self.center_ds_ratio = center_ds_ratio
        super().__init__(init_cfg=init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR repo, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        # self.bbox_assigner = TASK_UTILS.build(bbox_assigner)
        # self.dn_assigner = TASK_UTILS.build(dn_assigner)
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_center_cls = MODELS.build(loss_center_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        if loss_iou is not None:
            self.loss_iou = MODELS.build(loss_iou)
        else:
            self.loss_iou = None
        self.loss_seg = MODELS.build(loss_seg)
        self.loss_abu = MODELS.build(loss_abu)
        self.use_nms = use_nms
        self.class_wise_nms = class_wise_nms
        self.score_threshold = score_threshold
        self.test_nms = test_nms
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.neg_hard_num = neg_hard_num
        self.seg_neg_hard_num = neg_hard_num
        self.use_center = use_center
        self.decoder_embed_dims = decoder_embed_dims
        self.predict_segmentation = predict_segmentation
        self.predict_abundance = predict_abundance
        self.save_path = save_path
        if self.save_path is not None:
            os.makedirs(self.save_path,exist_ok=True)
        self.mask_threshold = mask_threshold
        self.mask_extend_pixel = mask_extend_pixel
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        # fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_cls = []
        for _ in range(2):
            fc_cls.append(Linear(self.embed_dims, self.embed_dims))
        fc_cls.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*fc_cls)
        self.cls_branch = fc_cls
        if self.predict_segmentation:
            fc_cls = []
            for _ in range(2):
                fc_cls.append(Linear(self.embed_dims, self.embed_dims))
            fc_cls.append(Linear(self.embed_dims, 1))
            fc_cls = nn.Sequential(*fc_cls)
            self.seg_branch = fc_cls
        if self.predict_abundance:
            fc_cls = []
            for _ in range(2):
                fc_cls.append(Linear(self.embed_dims, self.embed_dims))
            fc_cls.append(Linear(self.embed_dims, 1))
            fc_cls = nn.Sequential(*fc_cls)
            self.abu_branch = fc_cls

        reg_branch = []
        ratio=2
        reg_branch.append(Linear(self.decoder_embed_dims, self.decoder_embed_dims*ratio))
        reg_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs-1):
            reg_branch.append(Linear(self.decoder_embed_dims*ratio, self.decoder_embed_dims*ratio))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.decoder_embed_dims*ratio, 4))
        reg_branch = nn.Sequential(*reg_branch)
        if self.share_pred_layer:
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

        if self.use_center:
            center_cls = []
            for _ in range(2):
                center_cls.append(Linear(self.embed_dims, self.embed_dims))
            center_cls.append(Linear(self.embed_dims, 1))
            center_cls = nn.Sequential(*center_cls)
            self.center_branch = center_cls


    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            # for m in self.cls_branches:
            nn.init.constant_(self.cls_branch.bias, bias_init)
            if self.use_center:
                nn.init.constant_(self.center_branch.bias, bias_init)
            if self.predict_segmentation:
                nn.init.constant_(self.seg_branch.bias, bias_init)
            if self.predict_abundance:
                nn.init.constant_(self.abu_branch.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
            nn.init.constant_(m[-1].bias.data[2:], 0.0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)



    # def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
    #                         gt_instances: InstanceData,
    #                         img_meta: dict,
    #                         with_neg_cls:bool=True,
    #                         assigner_type:str = None) -> tuple:
    #     """Compute regression and classification targets for one image.
    #
    #     Outputs from a single decoder layer of a single feature level are used.
    #
    #     Args:
    #         cls_score (Tensor): Box score logits from a single decoder layer
    #             for one image. Shape [num_queries, cls_out_channels].
    #         bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
    #             for one image, with normalized coordinate (cx, cy, w, h) and
    #             shape [num_queries, 4].
    #         gt_instances (:obj:`InstanceData`): Ground truth of instance
    #             annotations. It should includes ``bboxes`` and ``labels``
    #             attributes.
    #         img_meta (dict): Meta information for one image.
    #
    #     Returns:
    #         tuple[Tensor]: a tuple containing the following for one image.
    #
    #         - labels (Tensor): Labels of each image.
    #         - label_weights (Tensor]): Label weights of each image.
    #         - bbox_targets (Tensor): BBox targets of each image.
    #         - bbox_weights (Tensor): BBox weights of each image.
    #         - pos_inds (Tensor): Sampled positive indices for each image.
    #         - neg_inds (Tensor): Sampled negative indices for each image.
    #     """
    #     img_h, img_w = img_meta['img_shape']
    #     factor = bbox_pred.new_tensor([img_w, img_h, img_w,
    #                                    img_h]).unsqueeze(0)
    #     num_bboxes = bbox_pred.size(0)
    #     # convert bbox_pred from xywh, normalized to xyxy, unnormalized
    #     bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
    #     bbox_pred = bbox_pred * factor
    #
    #     pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred,priors=bbox_pred)
    #     # assigner and sampler
    #     if assigner_type == 'dn':
    #         assign_result = self.dn_assigner.assign(
    #             pred_instances=pred_instances,
    #             gt_instances=gt_instances,
    #             img_meta=img_meta)
    #     else:
    #         assign_result = self.assigner.assign(
    #             pred_instances=pred_instances,
    #             gt_instances=gt_instances,
    #             img_meta=img_meta)
    #
    #     gt_bboxes = gt_instances.bboxes
    #     gt_labels = gt_instances.labels
    #     pos_inds = torch.nonzero(
    #         assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    #     neg_inds = torch.nonzero(
    #         assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
    #     pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
    #     pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]
    #     # label targets
    #     labels = gt_bboxes.new_full((num_bboxes,),
    #                                 self.num_classes,
    #                                 dtype=torch.long)
    #     labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
    #     label_weights = gt_bboxes.new_zeros(num_bboxes)
    #     label_weights[pos_inds] = 1
    #     label_weights[neg_inds] = 1
    #     if not with_neg_cls:
    #         label_weights[neg_inds] = 0
    #     # bbox targets
    #     bbox_targets = torch.zeros_like(bbox_pred)
    #     bbox_weights = torch.zeros_like(bbox_pred)
    #     bbox_weights[pos_inds] = 1.0
    #     # DETR regress the relative position of boxes (cxcywh) in the image.
    #     # Thus the learning target should be normalized by the image size, also
    #     # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
    #     pos_gt_bboxes_normalized = pos_gt_bboxes / factor
    #     pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
    #     bbox_targets[pos_inds] = pos_gt_bboxes_targets
    #     return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
    #             neg_inds)

    def _get_targets_single_center(self,
                                   center: Tensor,
                                   center_scores: Tensor,
                                   cls_scores: Tensor,
                                   spatial_shapes: Tensor,
                                   gt_instances: InstanceData,
                                   img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        feat_h = int(spatial_shapes[self.center_feat_indice][0]/self.center_ds_ratio)
        feat_w = int(spatial_shapes[self.center_feat_indice][1]/self.center_ds_ratio)
        factor = spatial_shapes.new_tensor([feat_w, feat_h]).unsqueeze(0)
        # factor = center.new_tensor([img_w, img_h,]).unsqueeze(0)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_cxcy = bbox_xyxy_to_cxcywh(gt_bboxes)[:, :2]
        gt_cxcy[:, 0] = gt_cxcy[:, 0] * feat_w / img_w
        gt_cxcy[:, 1] = gt_cxcy[:, 1] * feat_h / img_h
        gt_cxcy= gt_cxcy.long()
        gt_bboxes[:, 2:] -= 0.1
        gt_bboxes_x = gt_bboxes[:, 0::2]
        gt_bboxes_y = gt_bboxes[:, 1::2]
        gt_bboxes_x = torch.floor(gt_bboxes_x * feat_w / img_w)
        gt_bboxes_y = torch.floor(gt_bboxes_y * feat_h / img_h)
        gt_bboxes_x = gt_bboxes_x.long()
        gt_bboxes_y = gt_bboxes_y.long()
        heat_map = gt_bboxes.new_full((feat_h, feat_w), 0, dtype=torch.long)
        for t_i in range(gt_bboxes.size(0)):
            # if gt_bboxes_x[t_i, 1] - gt_bboxes_x[t_i, 0] > 3:
            #     gt_bboxes_x[t_i,1] = gt_cxcy[t_i, 0] + 1
            #     gt_bboxes_x[t_i,0] = gt_cxcy[t_i, 0] - 1
            # if gt_bboxes_y[t_i,1] - gt_bboxes_y[t_i,0] > 3:
            #     gt_bboxes_y[t_i,1] = gt_cxcy[t_i, 1] + 1
            #     gt_bboxes_y[t_i,0] = gt_cxcy[t_i, 1] - 1
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(gt_bboxes_y[t_i, 0], gt_bboxes_y[t_i, 1], gt_bboxes_y[t_i, 1]+1-gt_bboxes_y[t_i, 0],
                               dtype=torch.long, device=gt_cxcy.device),
                torch.linspace(gt_bboxes_x[t_i, 0], gt_bboxes_x[t_i, 1], gt_bboxes_x[t_i, 1]+1-gt_bboxes_x[t_i, 0],
                               dtype=torch.long, device=gt_cxcy.device))
            grid = torch.cat([grid_y.unsqueeze(-1), grid_x.unsqueeze(-1)], -1)
            grid = grid.view(-1, 2)
            value_input = gt_bboxes.new_full((grid.size(0),), -1, dtype=torch.long)
            heat_map.index_put_((grid[:,0],grid[:,1]), value_input)
            # cls_labels.index_put_((grid[:,0],grid[:,1]), value_input)
        value_input = gt_bboxes.new_full((gt_cxcy.size(0),), 1, dtype=torch.long)
        heat_map = heat_map.index_put_((gt_cxcy[:,1], gt_cxcy[:,0]), value_input)
        heat_map = heat_map.view(-1)
        mask = heat_map != -1
        # new_center_score = center_score[mask]
        # heat_map = heat_map[mask]
        pos_inds = torch.where(heat_map == 1)[0]
        ignore_inds = torch.where(heat_map == -1)[0]
        neg_inds = torch.where(heat_map == 0)[0]
        cls_labels = gt_bboxes.new_full((feat_h, feat_w), self.num_classes, dtype=torch.long)
        cls_labels = cls_labels.index_put_((gt_cxcy[:,1], gt_cxcy[:,0]), gt_labels)
        cls_labels = cls_labels.view(-1)
        center_labels = gt_bboxes.new_full((heat_map.size(0),), 1, dtype=torch.long)
        center_labels[pos_inds] = 0
        label_weights = gt_bboxes.new_ones(heat_map.size(0))
        if ignore_inds.numel() > 0:
            label_weights[ignore_inds] = 0
        if self.neg_hard_num>0:
            if self.use_center:
                _, indices = torch.sort(center_scores, dim=0, descending=True)
            else:
                cls_scores_max = torch.max(cls_scores, dim=-1, keepdim=True)[0]
                _, indices = torch.sort(cls_scores_max, dim=0, descending=True)
            sorted_inds = indices.squeeze()
            non_neg_inds = torch.cat([pos_inds,ignore_inds],dim=0)
            mask = torch.isin(sorted_inds, non_neg_inds)
            remaining_inds = sorted_inds[~mask]
            neg_hard_inds = remaining_inds[:self.neg_hard_num]
            new_inds = torch.cat([pos_inds, neg_hard_inds], dim=0)
            neg_inds = neg_hard_inds
            center_labels = center_labels[new_inds]
            cls_labels = cls_labels[new_inds]
            center_scores = center_scores[new_inds]
            cls_scores = cls_scores[new_inds]
            label_weights = gt_bboxes.new_ones(new_inds.size(0))
        return (center_labels, cls_labels, center_scores, cls_scores, label_weights, pos_inds, neg_inds)

    def _get_targets_single_pixel(self,
                                   seg_scores: Tensor,
                                   abu_scores: Optional[Tensor],
                                   spatial_shapes: Tensor,
                                   gt_seg: Tensor,
                                   gt_abu: Optional[Tensor],
                                   img_meta: dict) -> tuple:
        assert seg_scores.shape[0] == gt_seg.numel()
        assert spatial_shapes[0][0]*spatial_shapes[0][1] == gt_seg.numel()
        gt_seg = gt_seg.view(-1,1)
        pos_inds = torch.where(gt_seg >0)[0]
        seg_labels = gt_seg.detach().clone()
        seg_labels[gt_seg >0] = 0
        seg_labels[gt_seg == 0] = 1
        seg_labels = seg_labels.long()
        # gt_abu= gt_abu.view(-1,1)
        # pos_inds = torch.where(gt_abu > 0)[0]
        if self.seg_neg_hard_num== 0:
            neg_inds = torch.where(gt_seg == 0)[0]
            seg_label_weights = seg_scores.new_ones(seg_labels.size(0))
        else:
            seg_scores_max = torch.max(seg_scores, dim=-1, keepdim=True)[0]
            _, indices = torch.sort(seg_scores_max, dim=0, descending=True)
            sorted_inds = indices.squeeze()
            mask = torch.isin(sorted_inds, pos_inds)
            remaining_inds = sorted_inds[~mask]
            neg_hard_inds = remaining_inds[:self.seg_neg_hard_num]
            neg_inds = neg_hard_inds
            new_inds = torch.cat([pos_inds, neg_hard_inds], dim=0)
            seg_scores = seg_scores[new_inds]
            seg_labels = seg_labels[new_inds]
            seg_label_weights = seg_scores.new_ones(seg_labels.size(0))
        if self.predict_abundance:
            gt_abu = gt_abu.view(-1,1)
            abu_scores = abu_scores[pos_inds]
            abu_labels = gt_abu[pos_inds]*0.5+0.25
            abu_label_weights = seg_scores.new_ones(abu_labels.size(0))
        else:
            abu_scores = None
            abu_labels = None
            abu_label_weights = None
        return seg_scores, seg_labels, seg_label_weights,abu_scores, abu_labels, abu_label_weights, pos_inds, neg_inds


    # def _get_targets_single_bbox(self,
    #                         cls_score: Tensor,
    #                         bbox_pred: Tensor,
    #                         gt_instances: InstanceData,
    #                         img_meta: dict) -> tuple:
    #     """Compute regression and classification targets for one image.
    #
    #     Outputs from a single decoder layer of a single feature level are used.
    #
    #     Args:
    #         cls_score (Tensor): Box score logits from a single decoder layer
    #             for one image. Shape [num_queries, cls_out_channels].
    #         bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
    #             for one image, with normalized coordinate (cx, cy, w, h) and
    #             shape [num_queries, 4].
    #         gt_instances (:obj:`InstanceData`): Ground truth of instance
    #             annotations. It should includes ``bboxes`` and ``labels``
    #             attributes.
    #         img_meta (dict): Meta information for one image.
    #
    #     Returns:
    #         tuple[Tensor]: a tuple containing the following for one image.
    #
    #         - labels (Tensor): Labels of each image.
    #         - label_weights (Tensor]): Label weights of each image.
    #         - bbox_targets (Tensor): BBox targets of each image.
    #         - bbox_weights (Tensor): BBox weights of each image.
    #         - pos_inds (Tensor): Sampled positive indices for each image.
    #         - neg_inds (Tensor): Sampled negative indices for each image.
    #     """
    #     img_h, img_w = img_meta['img_shape']
    #     factor = bbox_pred.new_tensor([img_w, img_h, img_w,
    #                                    img_h]).unsqueeze(0)
    #     num_bboxes = bbox_pred.size(0)
    #     # convert bbox_pred from xywh, normalized to xyxy, unnormalized
    #     bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
    #     bbox_pred = bbox_pred * factor
    #
    #     pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred,priors=bbox_pred)
    #     # assigner and sampler
    #     assign_result = self.bbox_assigner.assign(
    #         pred_instances=pred_instances,
    #         gt_instances=gt_instances,
    #         img_meta=img_meta)
    #     gt_bboxes = gt_instances.bboxes
    #     gt_labels = gt_instances.labels
    #     pos_inds = torch.nonzero(
    #         assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    #     neg_inds = torch.nonzero(
    #         assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
    #     pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
    #     pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]
    #
    #     # bbox targets
    #     bbox_targets = torch.zeros_like(bbox_pred)
    #     bbox_weights = torch.zeros_like(bbox_pred)
    #     bbox_weights[pos_inds] = 1.0
    #
    #     # DETR regress the relative position of boxes (cxcywh) in the image.
    #     # Thus the learning target should be normalized by the image size, also
    #     # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
    #     pos_gt_bboxes_normalized = pos_gt_bboxes / factor
    #     pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
    #     bbox_targets[pos_inds] = pos_gt_bboxes_targets
    #     return (bbox_targets, bbox_weights, pos_inds, neg_inds)

    def loss_and_predict(
            self, hidden_states: Tuple[Tensor],
            batch_data_samples: SampleList) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples. Over-write because
        img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (tuple[Tensor]): Feature from the transformer
                decoder, has shape (num_decoder_layers, bs, num_queries, dim).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - predictions (list[:obj:`InstanceData`]): Detection
              results of each image after the post process.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
        outs = self(hidden_states)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas)
        return losses, predictions

    def forward(self, hidden_states: Tensor,
                references: List[Tensor],
                topk_cls_scores: Tensor,) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_coords = []
        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_coords.append(outputs_coord)
        all_layers_outputs_classes = topk_cls_scores.unsqueeze(0).repeat(hidden_states.shape[0],1,1,1)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss(self, hidden_states: Tensor,
             references: List[Tensor],
             centers: Tensor,
             center_scores: Tensor,
             topk_centers_scores: Tensor,
             cls_scores: Tensor,
             topk_cls_scores: Tensor,
             seg_scores:Optional[Tensor],
             abu_scores: Optional[Tensor],
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int],
             spatial_shapes: Tensor) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        batch_gt_seg = []
        batch_gt_abu = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if self.predict_segmentation:
                batch_gt_seg.append(data_sample.gt_pixel.seg)
            else:
                batch_gt_seg.append(None)
            if self.predict_abundance:
                batch_gt_abu.append(data_sample.gt_pixel.abu)
            else:
                batch_gt_abu.append(None)
        outs = self(hidden_states, references, topk_cls_scores)
        loss_inputs = outs + (centers, center_scores, topk_centers_scores, cls_scores, topk_cls_scores,
                              seg_scores,abu_scores,
                              batch_gt_instances, batch_img_metas, dn_meta, spatial_shapes,batch_gt_seg,batch_gt_abu)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
            self,
            all_layers_cls_scores: Tensor,
            all_layers_bbox_preds: Tensor,
            centers: Tensor,
            center_scores: Tensor,
            topk_centers_scores: Tensor,
            cls_scores: Tensor,
            topk_cls_scores: Tensor,
            seg_scores: Optional[Tensor],
            abu_scores: Optional[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            dn_meta: Dict[str, int],
            spatial_shapes: Tensor,
            batch_gt_seg: List,
            batch_gt_abu: List,
            batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        weight_bbox = 0
        weight_cls = 0
        # (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
        #  all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
        #     self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        # (matching_cls_scores, all_layers_matching_bbox_preds,
        #  denoising_cls_scores, all_layers_denoising_bbox_preds) = \
        #     self.split_outputsv1(cls_scores, all_layers_bbox_preds, dn_meta)
        loss_dict = dict()
        loss_center, loss_cls = self.loss_center(centers,
                                           center_scores,
                                           cls_scores,
                                           spatial_shapes,
                                           batch_gt_instances=batch_gt_instances,
                                           batch_img_metas=batch_img_metas)
        if self.use_center:
            loss_dict['loss_center'] = loss_center
        loss_dict['loss_cls'] = loss_cls
        if loss_cls <= self.loss_center_th:
            weight_bbox = 1
        if self.predict_segmentation:
            # seg_scores = seg_scores.sigmoid()
            if self.predict_abundance:
                abu_scores = abu_scores.sigmoid()
            loss_seg, loss_abu = self.loss_pixel(seg_scores, abu_scores, spatial_shapes,
                                                     batch_gt_seg,batch_gt_abu,
                                                 batch_img_metas=batch_img_metas)
            loss_dict['loss_seg'] = loss_seg*weight_bbox
            if self.predict_abundance:
                loss_dict['loss_abu'] = loss_abu*weight_bbox
        reg_targets = self.get_dn_targets(batch_gt_instances, batch_img_metas, dn_meta)
        dn_losses_bbox, dn_losses_iou = multi_apply(
            self._loss_dn_single,
            all_layers_bbox_preds,
            reg_targets=reg_targets,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)
        for num_dec_layer, (loss_bbox_i, loss_iou_i) in \
                enumerate(zip(dn_losses_bbox,  dn_losses_iou)):
            loss_dict[f'd{num_dec_layer+1}.dn_loss_bbox'] = loss_bbox_i*weight_bbox
            if self.loss_iou is not None:
                loss_dict[f'd{num_dec_layer+1}.dn_loss_iou'] = loss_iou_i*weight_bbox
        return loss_dict


    # def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
    #                         batch_gt_instances: InstanceList,
    #                         batch_img_metas: List[dict]) -> Tuple[Tensor]:
    #     """Loss function for outputs from a single decoder layer of a single
    #     feature level.
    #
    #     Args:
    #         cls_scores (Tensor): Box score logits from a single decoder layer
    #             for all images, has shape (bs, num_queries, cls_out_channels).
    #         bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
    #             for all images, with normalized coordinate (cx, cy, w, h) and
    #             shape (bs, num_queries, 4).
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #
    #     Returns:
    #         Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
    #         `loss_iou`.
    #     """
    #     num_imgs = cls_scores.size(0)
    #     cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    #     bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    #     cls_reg_targets = self.get_targets_bbox(cls_scores_list, bbox_preds_list,
    #                                        batch_gt_instances, batch_img_metas)
    #     (bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     bbox_targets = torch.cat(bbox_targets_list, 0)
    #     bbox_weights = torch.cat(bbox_weights_list, 0)
    #
    #
    #     # Compute the average number of gt boxes across all gpus, for
    #     # normalization purposes
    #     num_total_pos = bbox_targets.new_tensor([num_total_pos])
    #     num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
    #
    #     # construct factors used for rescale bboxes
    #     factors = []
    #     for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
    #         img_h, img_w, = img_meta['img_shape']
    #         factor = bbox_pred.new_tensor([img_w, img_h, img_w,
    #                                        img_h]).unsqueeze(0).repeat(
    #                                            bbox_pred.size(0), 1)
    #         factors.append(factor)
    #     factors = torch.cat(factors, 0)
    #
    #     # DETR regress the relative position of boxes (cxcywh) in the image,
    #     # thus the learning target is normalized by the image size. So here
    #     # we need to re-scale them for calculating IoU loss
    #     bbox_preds = bbox_preds.reshape(-1, 4)
    #     bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
    #     bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
    #
    #     # regression IoU loss, defaultly GIoU loss
    #     loss_iou = self.loss_iou(
    #         bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
    #
    #     # regression L1 loss
    #     loss_bbox = self.loss_bbox(
    #         bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
    #     return loss_bbox, loss_iou



    def loss_center(self,
                    centers: Tensor,
                    center_scores: Tensor,
                    cls_scores: Tensor,
                    spatial_shapes: Tensor,
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = centers.size(0)
        if center_scores is None:
            center_scores_list = [cls_scores[i] for i in range(num_imgs)]
        else:
            center_scores_list = [center_scores[i] for i in range(num_imgs)]

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        # center_scores =center_scores.view(-1, center_scores.shape[2])
        # cls_scores = cls_scores.view(-1, cls_scores.shape[2])
        centers_list = [centers[i] for i in range(num_imgs)]
        spatial_shapes_list = [spatial_shapes for i in range(num_imgs)]
        (center_labels_list, cls_labels_list, center_scores_list, cls_scores_list, label_weights_list,  pos_inds_list, neg_inds_list,) = multi_apply(self._get_targets_single_center,
                                centers_list, center_scores_list, cls_scores_list, spatial_shapes_list, batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        center_labels = torch.cat(center_labels_list, 0)
        cls_labels = torch.cat(cls_labels_list, 0)
        center_scores = torch.cat(center_scores_list, 0)
        cls_scores = torch.cat(cls_scores_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * 0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                centers.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        if self.use_center:
            loss_center = self.loss_center_cls(
                center_scores, center_labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_center = None
        loss_cls = self.loss_cls(
            cls_scores, cls_labels, label_weights, avg_factor=cls_avg_factor)
        return loss_center, loss_cls

    def loss_pixel(self,
                    seg_scores: Tensor,
                    abu_scores: Tensor,
                    spatial_shapes: Tensor,
                    batch_gt_seg: List,
                    batch_gt_abu: List,
                    batch_img_metas: List) -> Tuple[Tensor]:
        num_imgs = seg_scores.size(0)
        seg_scores_list = [seg_scores[i] for i in range(num_imgs)]
        if abu_scores is not None:
            abu_scores_list = [abu_scores[i] for i in range(num_imgs)]
        else:
            abu_scores_list = [None for i in range(num_imgs)]
        spatial_shapes_list = [spatial_shapes for i in range(num_imgs)]
        (seg_scores_list, seg_labels_list, seg_label_weights_list,abu_scores_list, abu_labels_list, abu_label_weights_list, pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single_pixel,
                                            seg_scores_list, abu_scores_list,
                                            spatial_shapes_list,
                                            batch_gt_seg,
                                            batch_gt_abu,
                                            batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        seg_scores = torch.cat(seg_scores_list, 0)
        seg_labels = torch.cat(seg_labels_list, 0)
        seg_label_weights = torch.cat(seg_label_weights_list, 0)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                seg_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_seg = self.loss_seg(
                    seg_scores, seg_labels, seg_label_weights, avg_factor=cls_avg_factor)
        if self.predict_abundance:
            abu_scores = torch.cat(abu_scores_list, 0)
            abu_labels = torch.cat(abu_labels_list, 0)
            abu_label_weights = torch.cat(abu_label_weights_list, 0)
            # cls_avg_factor = num_total_pos * 1.0
            # if self.sync_cls_avg_factor:
            #     cls_avg_factor = reduce_mean(
            #         seg_scores.new_tensor([cls_avg_factor]))
            # cls_avg_factor = max(cls_avg_factor, 1)
            loss_abu = self.loss_abu(
                abu_scores, abu_labels, abu_label_weights)
        else:
            loss_abu = None
        return loss_seg, loss_abu

    # def get_targets(self, cls_scores_list: List[Tensor],
    #                 bbox_preds_list: List[Tensor],
    #                 batch_gt_instances: InstanceList,
    #                 batch_img_metas: List[dict],
    #                 with_neg_cls:bool=True,
    #                 assigner_type:str = None) -> tuple:
    #     """Compute regression and classification targets for a batch image.
    #
    #     Outputs from a single decoder layer of a single feature level are used.
    #
    #     Args:
    #         cls_scores_list (list[Tensor]): Box score logits from a single
    #             decoder layer for each image, has shape [num_queries,
    #             cls_out_channels].
    #         bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
    #             decoder layer for each image, with normalized coordinate
    #             (cx, cy, w, h) and shape [num_queries, 4].
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #
    #     Returns:
    #         tuple: a tuple containing the following targets.
    #
    #         - labels_list (list[Tensor]): Labels for all images.
    #         - label_weights_list (list[Tensor]): Label weights for all images.
    #         - bbox_targets_list (list[Tensor]): BBox targets for all images.
    #         - bbox_weights_list (list[Tensor]): BBox weights for all images.
    #         - num_total_pos (int): Number of positive samples in all images.
    #         - num_total_neg (int): Number of negative samples in all images.
    #     """
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      pos_inds_list,
    #      neg_inds_list) = multi_apply(self._get_targets_single,
    #                                   cls_scores_list, bbox_preds_list,
    #                                   batch_gt_instances, batch_img_metas,
    #                                   with_neg_cls=with_neg_cls,
    #                                   assigner_type= assigner_type)
    #     num_total_pos = sum((inds.numel() for inds in pos_inds_list))
    #     num_total_neg = sum((inds.numel() for inds in neg_inds_list))
    #     if not with_neg_cls:
    #         num_total_neg = 0
    #     return (labels_list, label_weights_list, bbox_targets_list,
    #             bbox_weights_list, num_total_pos, num_total_neg)

    # def get_targets_bbox(self,
    #                 cls_scores_list: List[Tensor],
    #                 bbox_preds_list: List[Tensor],
    #                 batch_gt_instances: InstanceList,
    #                 batch_img_metas: List[dict]) -> tuple:
    #     """Compute regression and classification targets for a batch image.
    #
    #     Outputs from a single decoder layer of a single feature level are used.
    #
    #     Args:
    #         cls_scores_list (list[Tensor]): Box score logits from a single
    #             decoder layer for each image, has shape [num_queries,
    #             cls_out_channels].
    #         bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
    #             decoder layer for each image, with normalized coordinate
    #             (cx, cy, w, h) and shape [num_queries, 4].
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #
    #     Returns:
    #         tuple: a tuple containing the following targets.
    #
    #         - labels_list (list[Tensor]): Labels for all images.
    #         - label_weights_list (list[Tensor]): Label weights for all images.
    #         - bbox_targets_list (list[Tensor]): BBox targets for all images.
    #         - bbox_weights_list (list[Tensor]): BBox weights for all images.
    #         - num_total_pos (int): Number of positive samples in all images.
    #         - num_total_neg (int): Number of negative samples in all images.
    #     """
    #     (bbox_targets_list, bbox_weights_list, pos_inds_list,
    #      neg_inds_list) = multi_apply(self._get_targets_single_bbox, cls_scores_list,
    #                                   bbox_preds_list,
    #                                   batch_gt_instances, batch_img_metas)
    #     num_total_pos = sum((inds.numel() for inds in pos_inds_list))
    #     num_total_neg = sum((inds.numel() for inds in neg_inds_list))
    #     return (bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)


    def _loss_dn_single(self, dn_bbox_preds: Tensor,
                        reg_targets: Tuple[list, int],
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos) = reg_targets
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = dn_bbox_preds.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)
        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors
        # regression IoU loss, defaultly GIoU loss
        if self.loss_iou is not None:
            loss_iou = self.loss_iou(
                bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)
        else:
            loss_iou = None
        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_bbox, loss_iou

    def get_dn_targets(self, batch_gt_instances: InstanceList,
                       batch_img_metas: dict, dn_meta: Dict[str,
                                                            int]) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,pos_inds_list) = multi_apply(
             self._get_dn_targets_single,
             batch_gt_instances,
             batch_img_metas,
             dn_meta=dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,)

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        # label targets
        labels = gt_bboxes.new_full((num_denoising_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_zeros(num_denoising_queries)
        label_weights[pos_inds] = 1.0
        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights,bbox_targets, bbox_weights, pos_inds)


    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:,:, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)


    @staticmethod
    def split_outputsv1(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                centers: Tensor,
                center_scores: Tensor,
                topk_centers_scores: Tensor,
                cls_scores: Tensor,
                topk_cls_scores: Tensor,
                seg_scores: Optional[Tensor],
                abu_scores: Optional[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(hidden_states, references, topk_cls_scores)
        if self.predict_segmentation:
            seg_scores = seg_scores.sigmoid()
            if self.predict_abundance:
                abu_scores = torch.clamp((abu_scores.sigmoid()-0.25)*2,0,1)
        predictions = self.predict_by_feat(
            *outs,seg_scores,abu_scores, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        seg_scores: Optional[Tensor],
                        abu_scores: Optional[Tensor],
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            if self.predict_segmentation:
                seg_score = seg_scores[img_id]
            else:
                seg_score = None
            if self.predict_abundance:
                abu_score = abu_scores[img_id]
            else:
                abu_score = None
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   seg_score,abu_score,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                seg_score: Optional[Tensor],
                                abu_score: Optional[Tensor],
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))

        img_shape = img_meta['img_shape']
        assert self.loss_cls.use_sigmoid
        cls_score = cls_score.sigmoid()
        # scores, indexes = cls_score.view(-1).topk(max_per_img)
        scores, indexes = torch.sort(cls_score.view(-1), descending=True)
        # indexes = indexes[scores > self.score_threshold]
        # scores = scores[scores > self.score_threshold]
        det_labels = indexes % self.num_classes
        bbox_index = torch.div(indexes, self.num_classes, rounding_mode='trunc')
        bbox_pred = bbox_pred[bbox_index]
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if self.use_nms:
            if det_labels.numel() > 0:
                bboxes_scores, keep = batched_nms(det_bboxes, scores.contiguous(), det_labels, self.test_nms, class_agnostic=(not self.class_wise_nms))
                if keep.numel() > max_per_img:
                    bboxes_scores = bboxes_scores[:max_per_img]
                    det_labels = det_labels[keep][:max_per_img]
                else:
                    det_labels = det_labels[keep]
                det_bboxes = bboxes_scores[:, :-1]
                scores = bboxes_scores[:, -1]
        if self.pre_bboxes_round:
            det_bboxes = adjust_bbox_to_pixel(det_bboxes)
        if rescale:
            # assert img_meta.get('scale_factor') is not None
            # det_bboxes /= det_bboxes.new_tensor(
            #     img_meta['scale_factor']).repeat((1, 2))
            # rw by lzx
            if img_meta.get('scale_factor') is not None:
                det_bboxes /= det_bboxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, 2))

        if self.predict_segmentation:
            img_h, img_w = img_meta['ori_shape'][:2]
            seg_score = seg_score.view(img_h, img_w)
            seg_mask=seg_score>self.mask_threshold
            N = det_bboxes.size(0)
            im_mask = torch.zeros(
                N,
                img_shape[0],
                img_shape[1],
                device=det_bboxes.device,
                dtype=torch.bool)
            for i in range(N):
                x0, y0, x1, y1 = det_bboxes[i,:]
                x0 = max(int(x0)-self.mask_extend_pixel,0)
                x1 = min(int(x1)+self.mask_extend_pixel,img_w)
                y0 = max(int(y0)-self.mask_extend_pixel,0)
                y1 = min(int(y1)+self.mask_extend_pixel,img_h)
                im_mask[i, y0:y1,x0:x1] =seg_mask[y0:y1,x0:x1 ]
            new_mask = torch.sum(im_mask,dim=0)
            seg_score_new = seg_score.clone().detach()
            seg_score_new[new_mask==0] = 0
            if self.save_path is not None:
                img_name=img_meta['img_path'].split('/')[-1].replace('.npy','').replace('.png','').replace('.jpg','').replace('.mat','')
                sio.savemat(os.path.join(self.save_path,img_name+'_seg_scoremap.mat'),
                            {'data':seg_score.detach().cpu().numpy()})
                sio.savemat(os.path.join(self.save_path,img_name+'_seg_predictionmap.mat'),
                            {'data':seg_score_new.detach().cpu().numpy()})
                if self.predict_abundance:
                    abu_score = abu_score.view(img_h, img_w)
                    abu_score_new = abu_score.clone().detach()
                    abu_score_new[new_mask == 0] = 0
                    sio.savemat(os.path.join(self.save_path, img_name + '_abu_scoremap.mat'),
                                {'data': abu_score.detach().cpu().numpy()})
                    sio.savemat(os.path.join(self.save_path, img_name + '_abu_predictionmap.mat'),
                                {'data': seg_score_new.detach().cpu().numpy()})
        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        if self.predict_segmentation:
            results.masks = im_mask
        return results
