_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py'
]
model = dict(
    decode_head=dict(
        num_heads=1,
    )
)
