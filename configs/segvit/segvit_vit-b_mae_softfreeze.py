_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py'
]
checkpoint = './pretrained/vit_base_p16_mae.pth'
model = dict(
    pretrained=checkpoint,
)


optimizer = dict(_delete_=True, type='AdamW', lr=0.000002, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.),
                                                 'ln': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=100.),
                                                 }))