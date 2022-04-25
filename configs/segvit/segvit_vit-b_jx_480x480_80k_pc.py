_base_ = [
    '../_base_/models/seg_vit-b16.py',
    '../_base_/datasets/pascal_context.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
checkpoint = './pretrained/vit_base_p16_jx.pth'
out_indices = [5, 7, 11]
img_size = 480
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(img_size, img_size)
    ),
    decode_head=dict(
        img_size=img_size,
        num_classes=60,
        loss_decode=dict(
            type='ATMLoss', num_classes=60, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320))
)


data = dict(samples_per_gpu=1,)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.02,
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

