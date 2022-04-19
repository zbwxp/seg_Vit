_base_ = [
    '../segvit_vit-l_jax.py'
]
out_indices = [7, 11]
model = dict(
    backbone=dict(
        type='vit_shrink',
        shrink_idx=8,
        out_indices=out_indices,
    ),
    decode_head=dict(
        type="TPNHead",
        num_layers=3,
        use_stages=len(out_indices),
        loss_decode=dict(_delete_=True,
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    )
)
