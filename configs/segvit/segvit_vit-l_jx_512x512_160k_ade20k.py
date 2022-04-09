_base_ = [
    '../_base_/models/seg_vit-b16.py',
    '../_base_/datasets/ade20k_640x640.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
in_channels = 1024
checkpoint = './pretrained/vit_large_p16_jx.pth'
out_indices = [7, 15, 23]
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(640, 640),
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        out_indices=out_indices),
    decode_head=dict(
        in_channels=in_channels,
        channels=in_channels,
        embed_dims=in_channels // 2,
        num_heads=16,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608)),
)

data = dict(samples_per_gpu=1,)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
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
