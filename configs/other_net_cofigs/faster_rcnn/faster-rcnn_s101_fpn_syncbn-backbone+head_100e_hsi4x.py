_base_ = './faster-rcnn_s50_fpn_syncbn-backbone+head_100e_hsi4x.py'
model = dict(
    backbone=dict(
        stem_channels=128,
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://resnest101')))
