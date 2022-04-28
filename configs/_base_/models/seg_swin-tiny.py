# model settings
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_size = 512
in_channels = 768
out_indices = (1, 2, 3)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=out_indices,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    neck=dict(
        type='Deformable_DETR',
        in_channels=[192, 384, 768],
        num_outs=3,
    ),
    decode_head=dict(
        type="TPNATMHead_swin",
        img_size=img_size,
        in_channels=256,
        channels=in_channels,
        num_classes=150,
        num_layers=1,
        num_heads=8,
        use_proj=False,
        shrink_ratio=8,
        embed_dims=256,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=3, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)
find_unused_parameters = True