# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VideoParser',
        inplanes=64, 
        num_chs=(64, 128, 256, 512), 
        patch_sizes=[8, 7, 7, 7], 
        num_heads=[1, 2, 4, 8],
        num_enc_heads=[1, 2, 4, 8], 
        num_parts=[32, 32, 32, 32], 
        num_layers=[1, 1, 2, 1], 
        ffn_exp=3,
        has_last_encoder=False, 
        drop_path=0.1,
        local_attn='joint'),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg = dict(average_clips='prob')
)

