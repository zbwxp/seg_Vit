_base_ = [
    '../segvit_vit-b_jx_512x512_160k_ade20k.py'
]
out_indices = [5, 7, 11]
model = dict(
    backbone=dict(
        type='vit_shrink',
        shrink_idx=6,
        out_indices=out_indices,
    ),
    decode_head=dict(
        type="TPNATMROLLHead",
        num_layers=2,
        use_stages=len(out_indices),
        reverse=True,
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=len(out_indices), loss_weight=1.0),
    )
)
find_unused_parameters = False