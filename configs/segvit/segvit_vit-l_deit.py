_base_ = [
    './segvit_vit-l_jx_640x640_160k_ade20k.py',

]
in_channels = 1024
img_size = 640
out_indices = [7, 15, 23]
checkpoint = './pretrained/deit_3_large_384_21k.pth'
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type="deit_iii",
        img_size=img_size,
        pretrained=checkpoint,
        pretrained_21k=True,
    ),
    # decode_head=dict(
    #     img_size = img_size
    # ),
    # test_cfg=dict(mode='slide', crop_size=(320, 320), stride=(240, 240)),
)
data = dict(samples_per_gpu=4,)
