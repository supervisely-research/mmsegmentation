_base_ = './mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py'

# ============================================================================
# ARCHITECTURE
# ============================================================================
num_classes = 8
crop_size = (512, 512)

data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=num_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
    )
)

# ============================================================================
# DATA
# ============================================================================
data_root = '/root/tomatoes'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2560, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='TomatoDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='train_5shot_seed0/images/training',
            seg_map_path='train_5shot_seed0/annotations/training'
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='TomatoDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='val/images/validation',
            seg_map_path='val/annotations/validation'
        ),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# ============================================================================
# OPTIMIZER & SCHEDULER
# ============================================================================
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.00002,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=200
    ),
    # Main scheduler
    # dict(
    #     type='PolyLR',
    #     eta_min=0.0,
    #     power=1.0,
    #     by_epoch=False,
    #     begin=200,
    #     end=2000
    # )
]

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=10
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Metrics
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore']
)
test_evaluator = val_evaluator

# ============================================================================
# HOOKS & LOGGING
# ============================================================================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=3,
        save_best='mIoU'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False)
)

# TensorBoard logging
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# ============================================================================
# OUTPUT
# ============================================================================
work_dir = './work_dirs/mask2former_swin_tomato_5shot'

# Disable auto scale lr
auto_scale_lr = dict(enable=False)
