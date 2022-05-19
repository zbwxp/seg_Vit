_base_ = [
    '../segvit_vit-b_jx_512x512_160k_ade20k.py'
]
out_indices = [11]
model = dict(
    backbone=dict(
        type='vit_shrink',
        shrink_idx=6,
        out_indices=out_indices,
    ),
    decode_head=dict(
        type="ATMSingleHead",
        num_layers=3,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=150, dec_layers=3, loss_weight=1.0),
    )
)
data = dict(samples_per_gpu=4,)
find_unused_parameters = False