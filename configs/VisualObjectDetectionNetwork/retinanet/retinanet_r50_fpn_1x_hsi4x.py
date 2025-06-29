_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../datasets/hsi_detection4x.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta_hsi.py'
]
in_channels = 30
model = dict( #backbone=dict(init_cfg=dict(_delete_=True,type='Kaiming')),
            backbone=dict(
                in_channels=in_channels),
            bbox_head=dict(num_classes=8),
                )
# optimizer
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
