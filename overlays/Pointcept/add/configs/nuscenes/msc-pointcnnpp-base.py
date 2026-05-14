_base_ = ["../_base_/default_runtime.py"]
data_root = "/path/to/nuscenes_processed"

batch_size = 24
num_worker = 4
mix_prob = 0
empty_cache = False
enable_amp = False
evaluate = False
find_unused_parameters = True

model = dict(
    type="MSC-v1m2-pointcnnpp",
    backbone=dict(
        type="ResUNetPointCNNpp",
        in_channels=1,
        num_classes=0,
        base_channels=32,
        bn_momentum=0.05,
        normalize_feature=False,
        voxel_size=0.1,
        # channels format: (enc1, enc2, enc3, enc4, dec4, dec3, dec2, dec1)
        channels=(32, 64, 128, 320, 320, 128, 128, 128),
        # layers format: (enc0, enc1, enc2, enc3, dec3, dec2, dec1, dec0)
        layers=(2, 3, 4, 8, 2, 2, 2, 2),
        drop_path_rate=0.1,
    ),
    backbone_in_channels=1,
    backbone_out_channels=128,
    mask_grid_size=0.5,
    mask_rate=0.4,
    view1_mix_prob=0.0,
    view2_mix_prob=0.0,
    matching_max_k=8,
    matching_max_radius=1.2,
    matching_max_pair=8192,
    nce_t=0.4,
    contrast_weight=1,
    reconstruct_weight=0,
    use_boundary_aware_loss=False,
    reconstruct_color=False,
    reconstruct_normal=False,
    partitions=4,
    r1=1.0,
    r2=4.0,
    # Hard negative mining parameters
    use_hard_negative_mining=True,
    hard_neg_ratio=0.2,
    hard_neg_weight=1.4,
)

epoch = 5
eval_epoch = 5
optimizer = dict(type="SGD", lr=0.08, momentum=0.8, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

dataset_type = "NuScenesDataset"

data = dict(
    num_classes=16,
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split=["train"],
        data_root=data_root,
        sweeps=10,
        transform=[
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(
                type="ContrastiveViewsGenerator",
                view_keys=("coord", "strength", "origin_coord"),
                view_trans_cfg=[
                    dict(
                        type="Update",
                        keys_dict=dict(
                            index_valid_keys=[
                                "coord",
                                "color",
                                "normal",
                                "strength",
                                "segment",
                                "instance",
                                "origin_coord",
                            ]
                        ),
                    ),
                    dict(type="RandomRotate", angle=[-3.1416, 3.1416], axis="z", p=1),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.01, clip=0.05),
                    dict(type="GridSample", grid_size=0.1, hash_type="fnv", mode="train", return_grid_coord=True),
                ],
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "view1_origin_coord",
                    "view1_grid_coord",
                    "view1_coord",
                    "view1_strength",
                    "view2_origin_coord",
                    "view2_grid_coord",
                    "view2_coord",
                    "view2_strength",
                ),
                offset_keys_dict=dict(view1_offset="view1_coord", view2_offset="view2_coord"),
                view1_feat_keys=("view1_strength",),
                view2_feat_keys=("view2_strength",),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

