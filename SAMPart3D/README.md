# SAMPart3D: Segment Any Part in 3D Objects

## [Project Page](https://yhyang-myron.github.io/SAMPart3D-website/) | [Paper](https://arxiv.org/abs/2411.07184) | [Dataset: PartObjaverse-Tiny](PartObjaverse-Tiny/PartObjaverse-Tiny.md)

![](assets/teaser.png)

## 🔧 Setup

### Installation
Please refer to [INSTALL.md](INSTALL.md).

### Preparation for training

1. Download pretrained PTv3-object [https://huggingface.co/yhyang-myron/SAMPart3D/tree/main](https://huggingface.co/yhyang-myron/SAMPart3D/tree/main).

2. Data prepocessing

    We use Blender to render multi-view rgb and depth of the 3D glb mesh. First Install Blender:
    ```bash
    wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
    tar -xf blender-4.0.0-linux-x64.tar.xz
    rm blender-4.0.0-linux-x64.tar.xz
    ```
    Then render rgb and depth:
    ```bash
    cd tools
    ${PATH_TO_BLENDER} -b -P blender_render_16views.py ${MESH_PATH} ${TYPES} ${OUTPUT_PATH}
    ```   
    For example:
    ```bash
    blender-4.0.0-linux-x64/blender -b -P blender_render_16views.py mesh_root/knight.glb glb data_root/knight
    ```

## 🚀 Running SAMPart3D
### 1. Train
Change the rendering **data_root**, **mesh_root** and **backbone_weight_path** in `configs/sampart3d/sampart3d-trainmlp-render16views.py`.
```bash
SAMPart3D
|-- ckpt
    |-- ptv3-object.pth
|-- mesh_root
    |-- knight.glb
|-- data_root
    |-- knight
        |-- meta.json
        |-- render_0000.webp
        |-- depth_0000.exr
        ...
```

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
sh scripts/train.sh -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -o ${OBJECT_UID}
```
For example:
```bash
sh scripts/train.sh -g 1 -d sampart3d -c sampart3d-trainmlp-render16views -n knight -o knight
```

The mesh segmentation results will be saved in `exp/${DATASET_NAME}/${EXP_NAME}/resuls`, and the visualization of point clouds and meshes will be saved in `exp/${DATASET_NAME}/${EXP_NAME}/vis_pcd/`.

### 2. Test more scales with pretrained MLPs
After training, the ckpt of the target mesh will be saved in `exp/${DATASET_NAME}/${EXP_NAME}/model/`, if you want to try more scales, you can directly load the weight. And modify the **val_scales_list** in `exp/${DATASET_NAME}/${EXP_NAME}/config.py`.

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
sh scripts/eval.sh -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${WEIGHT_NAME}
```
For example:
```bash
sh scripts/eval.sh -g 1 -d sampart3d -n knight -w 5000
```

### 3. Highlight 3D segments on multi-view renderings
Set **render_dir**, **mesh_path**, **results_dir**, **save_dir** in `tools/highlight_parts.py`.
```bash
python tools/highlight_parts.py
```

## 📚 Dataset: PartObjaverse-Tiny
Please refer to [PartObjaverse-Tiny.md](PartObjaverse-Tiny/PartObjaverse-Tiny.md).

## Acknowledgement
SAMPart3D is inspired by the following repos: [garfield](https://github.com/chungmin99/garfield), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3), [Pointcept](https://github.com/Pointcept/Pointcept), [FeatUp](https://github.com/mhamilton723/FeatUp), [dinov2](https://github.com/facebookresearch/dinov2), [segment-anything](https://github.com/facebookresearch/segment-anything), [PartSLIP2](https://github.com/zyc00/PartSLIP2).

Many thanks to the authors for sharing their codes.

## Citation
If you find SAMPart3D useful in your project, please cite our work. :)
```
@article{yang2024sampart3d,
  title={SAMPart3D: Segment Any Part in 3D Objects},
  author={Yang, Yunhan and Huang, Yukun and Guo, Yuan-Chen and Lu, Liangjun and Wu, Xiaoyang and Lam, Edmund Y and Cao, Yan-Pei and Liu, Xihui},
  journal={arXiv preprint arXiv:2411.07184},
  year={2024}
}
```
