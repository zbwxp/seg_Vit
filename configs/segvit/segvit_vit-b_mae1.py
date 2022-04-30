_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py'
]
checkpoint = './pretrained/mae_pretrain_vit_base_mmcls.pth'
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        _delete_=True,
        type='MAE',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(5, 7, 11),
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=1.0),
)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructorNew',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# find_unused_parameters=True