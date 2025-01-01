_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/hsi_detection4x.py', '../_base_/default_runtime.py'
]
in_channels = 30
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        in_channels=in_channels,
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

# dataset settings

# optimizer
# optim_wrapper = dict(
#     optimizer=dict(lr=0.04),
#     paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True))

# learning policy
# max_epochs = 12
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]
# train_cfg = dict(max_epochs=max_epochs)

# cudnn_benchmark=True can accelerate fix-size training
env_cfg = dict(cudnn_benchmark=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)
