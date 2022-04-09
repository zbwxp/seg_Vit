_base_ = [
    './segvit_vit-l_jx_512x512_160k_ade20k.py'
]
model = dict(
    decode_head=dict(
        loss_decode=dict(
            mask_weight=5.0,
            dice_weight=5.0,
            cls_weight=2.0,
            use_point=True,
            ),
    )
)
