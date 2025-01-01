_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/hsi_detection4x.py',
]

in_channels = 30
model = dict( #backbone=dict(init_cfg=dict(_delete_=True,type='Kaiming')),
            backbone=dict(
                type='ResNet',
                in_channels=in_channels),
              roi_head=dict(bbox_head=dict(num_classes=8)),
                )
train_cfg = dict(val_interval=12)