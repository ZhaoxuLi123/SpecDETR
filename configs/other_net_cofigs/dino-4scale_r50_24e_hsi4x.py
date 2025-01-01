_base_ = './dino-4scale_r50_12e_hsi4x.py'
max_epochs = 24
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=24)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20],
        gamma=0.1)
]
# default_hooks = dict(
#     checkpoint=dict(type='CheckpointHook',interval=-1, by_epoch=False,save_best='auto'))