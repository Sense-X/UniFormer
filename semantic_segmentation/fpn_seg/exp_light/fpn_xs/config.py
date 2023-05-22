_base_ = [
    '../../configs/_base_/models/fpn_uniformer_light.py',
    '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='UniFormer_Light',
        depth=[3, 5, 9, 3], 
        conv_stem=True,
        prune_ratio=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        trade_off=[[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        embed_dim=[64, 128, 256, 512], 
        head_dim=32, 
        mlp_ratio=[3, 3, 3, 3], 
        drop_path_rate=0.1),
    neck=dict(in_channels=[64, 128, 256, 512]),
    decode_head=dict(num_classes=150))


gpu_multiples=2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-8)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
