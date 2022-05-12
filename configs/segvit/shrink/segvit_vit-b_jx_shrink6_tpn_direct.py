_base_ = [
    '../segvit_vit-b_jx_512x512_160k_ade20k.py'
]
out_indices = [5, 11]
model = dict(
    backbone=dict(
        type='vit_shrink_direct',
        shrink_idx=6,
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
data = dict(samples_per_gpu=4,)
find_unused_parameters = False