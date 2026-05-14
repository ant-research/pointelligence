_base_ = ["../_base_/default_runtime.py"]
batch_size = 48
mix_prob = 0
empty_cache = False
enable_amp = True
class_weights = [
    1.2,
    3.0,
    1.0,
    1.0,
    1.5,
    2.5,
    1.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.8,
    1.2,
    1.1,
    1.0,
    1.0,
]
focal_gamma = 1.35
focal_alpha = [
    1.2*0.25,
    3.0*0.25,
    1.0*0.25,
    1.0*0.25,
    1.5*0.25,
    2.5*0.25,
    1.0*0.25,
    2.0*0.25,
    1.0*0.25,
    1.0*0.25,
    1.0*0.25,
    1.8*0.25,
    1.2*0.25,
    1.1*0.25,
    1.0*0.25,
    1.0*0.25,
]
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="ResUNetPointCNNpp",
        in_channels=1,
        num_classes=16,
        channels=(32, 64, 128, 320, 320, 128, 128, 128),
        layers=(2, 3, 4, 8, 2, 2, 2, 2),
        use_pos_encoding=False,
        normalize_feature=False,
        voxel_size=0.1,
        pre_downsample_grid_size=None,
        pos_fusion="concat",
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss", 
            weight=class_weights,
            loss_weight=0.35,
            ignore_index=-1
        ),
        dict(
            type="FocalLoss",
            gamma=focal_gamma,
            alpha=focal_alpha,
            reduction="mean",
            loss_weight=0.25,
            ignore_index=-1
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=0.4,
            ignore_index=-1
        ),
    ],
)
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.0018, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
dataset_type = "NuScenesDataset"
data_root = "/path/to/nuscenes_processed"
ignore_index = -1
names = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]
data = dict(
    num_classes=16,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        use_class_balanced_sampler=True,
        sampler_config=dict(
            num_classes=16,
            iou_levels=dict(
                very_low=dict(
                    classes=[1],
                    multiplier=1.3,
                ),
                low=dict(
                    classes=[4, 7],
                    multiplier=1.2, 
                ),
                medium=dict(
                    classes=[8, 11, 12, 13],
                    multiplier=1.1,
                ),
                high=dict(
                    classes=[0, 5, 6],
                    multiplier=1.0,
                ),
                very_high=dict(
                    classes=[2, 3, 9, 10, 14, 15],
                    multiplier=1.0,
                ),
            ),
            high_freq_threshold=0.15,
            high_freq_multiplier=0.6,
            use_relative_adjustment=True,
            target_samples_per_class=None,
            cache_dir="cache/class_indices_nuscenes",
            shuffle=True,
            seed=42,
        ),
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(
                type="Update",
                keys_dict={
                    "index_valid_keys": [
                        "coord",
                        "color",
                        "normal",
                        "strength",
                        "segment",
                        "instance",
                    ]
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("strength",),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
