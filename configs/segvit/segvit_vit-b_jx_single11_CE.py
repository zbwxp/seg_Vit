_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py'
]
out_indices = [11]
model = dict(
    backbone=dict(
        out_indices=out_indices,
    ),
    decode_head=dict(
        use_stages=len(out_indices),
        CE_loss=True,
        loss_decode=dict(_delete_=True,
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
)
