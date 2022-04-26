_base_ = [
    '../../_base_/models/seg_swin-tiny.py',
    '../../_base_/datasets/ade20k_640x640.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
checkpoint = './pretrained/swin_large.pth'
img_size = 640
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        window_size=12,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
    ),
    neck=dict(
        in_channels=[384, 768, 1536],
    ),
    decode_head=dict(
        img_size=img_size,
        num_layers=2,
    ),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608)),
)
data = dict(samples_per_gpu=1,)