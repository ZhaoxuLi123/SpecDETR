_base_ = 'retinanet_pvtv2-b0_fpn_100e_hsi4x.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        mlp_ratios=(4, 4, 4, 4),
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b5.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))

