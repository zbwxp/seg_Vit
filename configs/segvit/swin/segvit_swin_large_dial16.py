_base_ = [
    '../../_base_/models/seg_swin-tiny.py',
    '../../_base_/datasets/ade20k_640x640.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
checkpoint = './pretrained/swin_large.pth'
img_size = 640
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        window_size=12,
        depths=[2, 2, 18, 2],
        strides=(4, 2, 2, 1),
        num_heads=[6, 12, 24, 48],
    ),
    neck=dict(
        in_channels=[384, 768, 1536],
        out_channels=384,
        nhead=12,
    ),
    decode_head=dict(
        type="ATMHead",
        img_size=img_size,
        in_channels=384,
        embed_dims=384,
        num_layers=3,
        use_proj=False,
    ),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608)),
)
data = dict(samples_per_gpu=4,)
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                 'linear': dict(decay_mult=0.),
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
find_unused_parameters = False