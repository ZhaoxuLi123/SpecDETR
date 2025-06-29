_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/hsi_detection4x.py',
    '../_base_/schedules/schedule_100e.py', '../_base_/default_runtime.py'
]
in_channels = 30
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        in_channels=in_channels,
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b0.pth')),
    neck=dict(in_channels=[32, 64, 160, 256]))
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001))
