_base_ = [
    '../segvit_vit-l_jx_512x512_160k_ade20k.py'
]
model = dict(
    backbone=dict(
        type='vit_crop',
    ),
    decode_head=dict(
        crop_train=True,
    )
)
