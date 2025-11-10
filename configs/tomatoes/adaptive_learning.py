_base_ = './L_5shot.py'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='AdaptiveTomatoDataset',
        data_root='/root/tomatoes',
        data_prefix=dict(
            img_path='train_5shot_seed0/images/training',
            seg_map_path='train_5shot_seed0/annotations/training'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='RandomResize',
                scale=(2560, 512),
                ratio_range=(0.5, 2.0),
                keep_ratio=True
            ),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ],
        # Adaptive parameters
        max_samples=5,
        initial_samples=2,
        samples_per_stage=1
    )
)

custom_hooks = [
    dict(
        type='AdaptiveLearningHook',
        iters_per_stage=400,  # Add 1 sample every 400 iterations
        save_checkpoint=True
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=10
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=100,
        max_keep_ckpts=5,
        save_best='mIoU'
    )
)

work_dir = './work_dirs/mask2former_swin_tomato_adaptive'
