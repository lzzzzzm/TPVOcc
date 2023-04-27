_base_ = ['mmdet3d::_base_/default_runtime.py']

custom_imports = dict(imports=['projects.models',
                               'projects.datasets',
                               'projects.engine',
                               'projects.evaluation',], allow_failed_imports=False)

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], bgr_to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

num_classes = 18
bev_h = 200
bev_w = 200
embed_dims = 256
position_dims = 128
ffn_dims = 512
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_s_fpn_3x_coco.pth'
model = dict(
    type='TPVOcc',
    use_grid_mask=True,
    video_test_mode=True,
    data_preprocessor=dict(
        type='OccDataPreprocessor', **img_norm_cfg, pad_size_divisor=32),
    img_backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=80,
        depths=[4, 4, 21, 4],
        groups=[5, 10, 20, 40],
        mlp_ratio=4.,
        drop_path_rate=0.3,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=True,
        with_cp=True,
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[160, 320, 640],
        out_channels=embed_dims,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='TPVOccHead',
        pc_range=point_cloud_range,
        bev_h=bev_h,
        bev_w=bev_w,
        num_classes=num_classes,
        transformer=dict(
            type='TPVOccTransformer',
            pillar_h=16,
            num_classes=18,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            use_3d=True,
            use_conv=False,
            embed_dims=embed_dims,
            encoder=dict(
                type='TPVOccEncoder',
                num_layers=4,
                pc_range=point_cloud_range,
                num_points_in_pillar=8,
                transformerlayers=dict(
                    type='TPVOccTransformerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=embed_dims,
                            num_levels=1
                        ),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            embed_dims=embed_dims,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=embed_dims,
                                num_points=8
                            )
                        )
                    ],
                    feedforward_channels=ffn_dims,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                )
            )
        ),
        positional_encoding=dict(
            type='mmdet.LearnedPositionalEncoding',
            num_feats=position_dims,
            row_num_embed=bev_h,
            col_num_embed=bev_w
        ),
        loss_occ=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.8
        ),
        loss_lovasz=dict(
            type='CustomLovaszLoss',
            loss_weight=0.2,
            per_sample=True,
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        )

    )
)

dataset_type='NuScenesOcc'
data_root = 'data/occ3d-nus/'
test_transforms = [
    dict(
        type='RandomResize3D',
        scale=(1600, 900),
        ratio_range=(1., 1.),
        keep_ratio=True)
]
train_transforms = [dict(type='PhotoMetricDistortion3D')] + test_transforms
backend_args = None
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadOccAnnotations3D',
        with_occ_semantics=True,
        with_camera_mask=False,
        with_lidar_mask=False),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ResizeMultiViewImage',
         size=(128, 128)),
    dict(type='CustomPack3DDetInputs', keys=['img', 'occ_semantics'])
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=6,
        backend_args=backend_args),
    dict(type='MultiViewWrapper', transforms=test_transforms),
    dict(type='CustomPack3DDetInputs', keys=['img'])
]

metainfo = dict(classes=class_names)
data_prefix = dict(
    pts='',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        load_type='frame_based',
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='occ_infos_temporal_val.pkl',
        load_type='frame_based',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader
val_evaluator = dict(
    type='NuScenesOccMetric',
    data_root=data_root,
    ann_file=data_root + 'occ_infos_temporal_val.pkl',
    metric='MIoU',
    backend_args=backend_args)
test_evaluator = val_evaluator
optim_wrapper = dict(
    type='OptimWrapper',
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg={
        'layer_decay_rate': 1.0,
        'num_layers': 33,
        'depths': [4, 4, 21, 4]
    },
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
)


# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=24,
        T_max=24,
        eta_min_ratio=1e-3)
]

total_epochs = 24

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1, save_last=True))

load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

# setuptools 65 downgrades to 58.
# In mmlab-node we use setuptools 61 but occurs NO errors
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

