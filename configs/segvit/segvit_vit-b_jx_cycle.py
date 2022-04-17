_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py'
]
out_indices = [5, 7, 11]
model = dict(
    backbone=dict(
        out_indices=out_indices,
    ),
    decode_head=dict(
        type="ATMHead_cycle",
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=len(out_indices)+1, loss_weight=1.0),
    )
)
