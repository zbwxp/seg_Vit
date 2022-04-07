norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 768
out_indices = [3, 7, 11]
model = dict(
    type='EncoderDecoder',
    pretrained='./pretrained/mae_pretrain_vit_base.pth',
    # download the pretraining ViT-Base model: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
    backbone=dict(
        type='MAE',
        img_size=512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=True, # here different
        use_rel_pos_bias=True,
        init_values=1.,
        drop_path_rate=0.1,
        out_indices=out_indices
    ),
    decode_head=dict(
        type='ATMHead',
        img_size=img_size,
        in_channels=in_channels,
        channels=in_channels,
        embed_dims=in_channels // 2,
        num_classes=150,
        num_layers=3,
        num_heads=12,
        use_stages=len(out_indices),
        # loss_decode=dict(
        #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=3, loss_weight=1.0),
    ),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)
find_unused_parameters=True