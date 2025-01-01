_base_ = [
    './datasets/hsi_detection.py', '../_base_/default_runtime.py'
]

# fp16 = dict(loss_scale=512.)
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
norm = 'LN'  #'IN1d' 'LN''BN1d'
num_levels = 2
in_channels = 30
embed_dims = 256  # embed_dims256
model = dict(
    type='SpecDetr',
    num_queries = 900,  # num_matching_queries 900
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=num_levels,
    candidate_bboxes_size = 0.01, #  initial candidate_bboxes after encode 0.01
    scale_gt_bboxes_size = 0,  # [0,0.5)  0.25,
    training_dn = True,  #  use dn when training
    dn_only_pos = True,
    remove_last_candidate = True,  # when the last feacture size of backbone is 1
    data_preprocessor=dict(
        type='HSIDetDataPreprocessor'),
    backbone=dict(
        type='No_backbone_ST',
        in_channels=in_channels,
        embed_dims=embed_dims,
        patch_size=(1,),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        num_levels=num_levels,
        norm_cfg=dict(type=norm),
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=embed_dims, num_levels=num_levels, num_points=4, # local_attn_type ='fix_same_orientation',
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=embed_dims,
                feedforward_channels=embed_dims*8,  # 1024 for DeformDETR
                ffn_drop=0.0),
            norm_cfg=dict(type=norm),)),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=embed_dims, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=embed_dims, num_levels=num_levels, num_points=4, #local_attn_type = 'fix_same_orientation',  # fix_same_orientation
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=embed_dims,
                feedforward_channels=embed_dims*8,  # 1024 for DeformDETR 2048 for dino
                ffn_drop=0.0),
            norm_cfg=dict(type=norm),),  # 0.1 for DeformDETR  norm_cfg=dict(type='LN')
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=embed_dims//2,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='SpecDetrHead',
        num_classes=8,
        sync_cls_avg_factor=True,
        pre_bboxes_round = True,
        use_nms = True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,  # 0.5
        box_noise_scale=1,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100),
        # group_cfg=dict(dynamic=False, num_groups=10,
        #                num_dn_queries=None),
            ),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='DynamicHungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ],
            anomaly_factor = 5,
            match_nums = 8,
            normal_outlier = True,
            dynamic_match = True)),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=12)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# train_dataloader = dict(
#     batch_size=4,)
# test_dataloader = dict(
#     batch_size=4,)
#
# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # USER SHOULD NOT CHANGE ITS VALUES.
# # base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=4)