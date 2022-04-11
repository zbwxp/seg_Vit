_base_ = [
    './segvit_vit-b_jax.py'
]
out_indices = [2, 5, 8, 11]
model = dict(
    backbone=dict(
        out_indices=out_indices,
    ),
    decode_head=dict(
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=len(out_indices), loss_weight=1.0),
    )
)
