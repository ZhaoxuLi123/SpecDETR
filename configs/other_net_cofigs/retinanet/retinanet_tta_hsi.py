tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))

img_scales = [3,4,5,6]


tta_pipeline = [
    dict(type='LoadHyperspectralImageFromFiles', to_float32 =True, normalized_basis=3000),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='HSIResize', scale_factor=s, keep_ratio=True) for s in img_scales
        ], [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [dict(type='LoadAnnotations', with_bbox=True)],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ]])
]
