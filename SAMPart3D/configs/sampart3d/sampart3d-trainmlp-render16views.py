_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # dummy value
num_worker = 1  # dummy value
mix_prob = 0  # dummy value
empty_cache = False
enable_amp = True

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]

# model settings
model = dict(
    type="SAMPart3D",
    backbone_dim=384,
    output_dim=384,
    pcd_feat_dim=9,
    freeze_backbone=True,
    max_grouping_scale=2,
    use_hierarchy_losses=True,
    backbone = dict(
        type="PTv3-obj",
        in_channels=9,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(),
        enc_depths=(3, 3, 3, 6, 16),
        enc_channels=(32, 64, 128, 256, 384),
        enc_num_head=(2, 4, 8, 16, 24),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False)
)

# scheduler settings
epoch = 5000  # can be smaller
eval_epoch = 5000  # can be smaller
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=1e-8)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[1e-4],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10.0,
)

# dataset settings
dataset_type = "SAMPart3DDataset16Views"
data_root = ""
mesh_root = ""
backbone_weight_path = ""

# eval
val_scales_list = [0.0, 0.5, 1.0, 1.5, 2.0]
mesh_voting = True

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        mesh_root=mesh_root,
        sample_num=15000,
        pixels_per_image=256,
        batch_size=90,
        extent_scale=10.0,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.01,
                keys=("coord", "color", "normal", "origin_coord", "face_index"),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "inverse", "origin_coord", "face_index"),
                feat_keys=("coord", "normal", "color"),
            ),
        ],
    ),
)