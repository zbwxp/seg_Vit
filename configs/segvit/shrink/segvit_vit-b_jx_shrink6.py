_base_ = [
    '../segvit_vit-b_jx_512x512_160k_ade20k.py'
]
model = dict(
    backbone=dict(
        type='vit_shrink',
        shrink_idx=6,
    ),
)
