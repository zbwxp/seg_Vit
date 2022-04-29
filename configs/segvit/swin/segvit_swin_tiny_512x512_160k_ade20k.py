_base_ = [
    '../../_base_/models/seg_swin-tiny.py',
    '../../_base_/datasets/ade20k.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
checkpoint = './pretrained/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
model = dict(
    pretrained=checkpoint,
    neck=dict(
        out_channels=256,
    ),
    decode_head=dict(
        type="ATMHead",
        num_layers=3,
        use_proj=False,
    ),
)

data = dict(samples_per_gpu=4,)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                 'ln': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 }))
#
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

