_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py'
]
model = dict(
    decode_head=dict(
        num_layers=1,
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=1, loss_weight=1.0),
    )
)
