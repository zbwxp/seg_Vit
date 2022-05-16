_base_ = [
    '../segvit_vit-l_jax.py'
]
out_indices = [7, 23]
model = dict(
    backbone=dict(
        type='vit_shrink_conv',
        shrink_idx=8,
        out_indices=out_indices,
    ),
    decode_head=dict(
        type="TPNATMHead",
        num_layers=3,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=3, loss_weight=1.0),
    )
)
data = dict(
    samples_per_gpu=4,)