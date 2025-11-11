_base_ = '../mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py'

crop_size = (512, 512)

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

data_root = '/root/data'
train_dir = 'images/train'
val_dir = 'images/val'

classes = ['background', 'Core', 'Locule', 'Navel',
           'Pericarp', 'Placenta', 'Septum', 'Tomato']
num_classes = len(classes)
metainfo = None

data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=None),
    decode_head=dict(
        num_classes=num_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
    )
)

# pretrained = None
# load_from = '/weights/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth'

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
        data_prefix=dict(
            img_path='images/train/images/training',
            seg_map_path='images/train/annotations/training'
        ),
        pipeline=train_pipeline,
        reduce_zero_label=False
    )
)

val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

max_epochs = 100000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=200),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_epochs, val_interval=10)
val_cfg = None
test_cfg = None

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=200, max_keep_ckpts=5, save_best='auto'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')