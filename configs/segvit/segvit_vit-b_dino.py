_base_ = [
    './segvit_vit-b_jx_512x512_160k_ade20k.py',
]
checkpoint = './pretrained/vit_base_p16_dino.pth'
model = dict(
    pretrained=checkpoint,
    decode_head=dict(
        loss_decode=dict(
            cls_weight=5.0,
            ),
    )
)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)