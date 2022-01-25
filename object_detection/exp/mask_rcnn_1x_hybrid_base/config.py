_base_ = [
    '../../configs/_base_/models/mask_rcnn_uniformer_fpn.py',
    '../../configs/_base_/datasets/coco_instance.py',
    '../../configs/_base_/schedules/schedule_1x.py', 
    '../../configs/_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=[64, 128, 320, 512],
        layers=[5, 8, 20, 7],
        head_dim=64,
        drop_path_rate=0.3,
        use_checkpoint=False,
        windows=False,
        hybrid=True,
        window_size=14
    ),
    neck=dict(in_channels=[64, 128, 320, 512]))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
