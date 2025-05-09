# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import numpy as np
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class DynamicHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def __init__(
        self,
            match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict],
            base_match_num: int  = 1,
            match_num: int = 1,
            base_anomaly_factor: float = -1,
            anomaly_factor: float = 3,
            total_num_max: int  = 300,
            dynamic_match: bool = True,
            normal_outlier: bool = True
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'
        self.match_num = match_num
        self.dynamic_match = dynamic_match
        self.anomaly_factor= anomaly_factor
        self.base_match_num = base_match_num
        self.base_anomaly_factor = base_anomaly_factor
        self.total_num_max = total_num_max
        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        self.normal_outlier = normal_outlier
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)  # 先求900query与每个gt的cost，不同类型的cost相加

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        # lzx
        match_preds_max =  self.total_num_max #int(num_preds/3)
        matched_row_inds_all = np.empty((0,), dtype=np.int64)
        matched_col_inds_all = np.empty((0,), dtype=np.int64)
        cost_max = torch.max(cost)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds_all = np.concatenate((matched_row_inds_all, matched_row_inds))
        matched_col_inds_all = np.concatenate((matched_col_inds_all, matched_col_inds))
        th0 = None
        if self.match_num>1:
            if self.normal_outlier:
                th = torch.mean(cost, axis=0) - self.anomaly_factor * torch.std(cost, axis=0)
                # th = torch.mean(cost) - self.anomaly_factor * torch.std(cost)
                th = th.detach().cpu().numpy()
                if self.base_anomaly_factor>=0:
                        # th0 = torch.mean(cost, axis=0) - self.base_anomaly_factor * torch.std(cost, axis=0)
                        th0 = torch.mean(cost) - self.base_anomaly_factor * torch.std(cost)
                        th0 = th0.detach().cpu().numpy()
            else:
                q1 = np.percentile(cost, 25, axis=0)
                q3 = np.percentile(cost, 75, axis=0)
                iqr = q3 - q1
                th = q1 - self.anomaly_factor * iqr
                if self.base_anomaly_factor>0:
                        th0 = q1 - self.base_anomaly_factor * iqr
            # has_positive = np.any(th > 0)
        for ii in range(2, self.match_num+1):
            if len(matched_col_inds_all) < match_preds_max:
                cost[matched_row_inds,:] = cost_max+1
                matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
                if self.dynamic_match:
                    if ii<=self.base_match_num and ii > 1 and th0 is not None :
                        if th0 is not None :
                            matched_costs = cost[matched_row_inds, matched_col_inds]
                            filtered_inds = np.where(matched_costs.detach().cpu().numpy() < th0)
                            matched_row_inds = matched_row_inds[filtered_inds]
                            matched_col_inds = matched_col_inds[filtered_inds]
                    else:
                            matched_costs = cost[matched_row_inds, matched_col_inds]
                            filtered_inds = np.where(matched_costs.detach().cpu().numpy() < th)
                            # if len(filtered_inds[0])>0:
                            #     print(ii,"/",self.match_num, "object",len(matched_row_inds),'bbox',len(filtered_inds[0]))
                            # 根据筛选后的索引获取对应的matched_row_inds和matched_col_inds
                            matched_row_inds = matched_row_inds[filtered_inds]
                            matched_col_inds = matched_col_inds[filtered_inds]
                    if len(matched_row_inds) == 0:
                        break
                if (len(matched_row_inds)+len(matched_col_inds_all))>match_preds_max:
                    need_inds_num = len(matched_row_inds)+len(matched_col_inds_all)-match_preds_max
                    matched_row_inds = matched_row_inds[:need_inds_num]
                    matched_col_inds = matched_col_inds[:need_inds_num]
                matched_row_inds_all = np.concatenate((matched_row_inds_all, matched_row_inds))
                matched_col_inds_all = np.concatenate((matched_col_inds_all, matched_col_inds))
        matched_row_inds = torch.from_numpy(matched_row_inds_all).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds_all).to(device)
        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
