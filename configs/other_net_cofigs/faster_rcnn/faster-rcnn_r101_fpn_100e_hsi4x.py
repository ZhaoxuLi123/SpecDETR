_base_ = './faster-rcnn_r50_fpn_100e_hsi4x.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
