_base_ = '../mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py'

# Image and crop settings
crop_size = (1024, 1024)

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# Data preprocessor
data_preprocessor = dict(size=crop_size)

# Paths (will be overridden in main.py)
data_root = ''
classes = None
num_classes = None
metainfo = None

# Model configuration
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=None),
    decode_head=dict(num_classes=num_classes)
)

# Pretrained weights from Docker image
load_from = '/weights/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth'

# Dataset configuration
train_dataloader = dict(
    batch_size=2,
    persistent_workers=False,
    num_workers=0,
    sampler=dict(type='CustomSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='OnlineTrainingDataset',
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img_path='', seg_map_path=''),
        pipeline=train_pipeline,
        reduce_zero_label=False
    )
)

# Disable validation and testing
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

# Optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

# Training schedule
max_iters = 100000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=200),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=10)
val_cfg = None
test_cfg = None

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

# Visualization
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)