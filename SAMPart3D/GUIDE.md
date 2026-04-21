# Running SAMPart3D Inference (example: `knight.glb`)

## 1. Render multi-view data with Blender
```bash
# dependencies: Blender >= 4.0 downloaded under ~/Downloads
cd /home/ehud/code/python/brickgpt/SAMPart3D
/home/ehud/Downloads/blender-4.2.16-linux-x64/blender \
  -b -P tools/blender_render_16views.py \
  mesh_root/knight.glb glb data/knight
```

## 2. Fit the per-object MLP (uses PTv3 backbone features)
```bash
cd /home/ehud/code/python/brickgpt/SAMPart3D
conda activate sampart3d     # environment prepared via INSTALL.md
scripts/train.sh \
  -g 1 -d sampart3d \
  -c sampart3d-trainmlp-render16views_knight \
  -n knight -o knight
```
Outputs appear under `exp/sampart3d/knight/` (MLP checkpoint + colored meshes).

## 3. Visualize part segments
### Option A – Open colored meshes in MeshLab/Blender
```bash
meshlab exp/sampart3d/knight/vis_pcd/last/mesh_1.0.ply
```
Each scale (`mesh_0.0.ply`, `mesh_0.5.ply`, …) renders the knight with faces colored by the predicted part IDs.

### Option B – Overlay predictions on the renders
```bash
python tools/highlight_parts.py \
  --render-dir data/knight \
  --mesh-path mesh_root/knight.glb \
  --results-dir exp/sampart3d/knight/results/last \
  --save-dir exp/sampart3d/knight/render_highlights
```

---

# What each stage does

## Rendering
- `tools/blender_render_16views.py` normalizes the mesh, samples 16 fixed camera poses, and writes multi-view RGB (`render_XXXX.webp`), depth (`depth_XXXX.exr`), and `meta.json`.  
- This step ensures every render has known camera intrinsics/extrinsics, mesh scaling, and depth so that later we can project image pixels back to mesh coordinates (“lifting”).

## MLP training (per object)
- `scripts/train.sh` wraps `launch/train.py`, which loads the PTv3 backbone weights (frozen) and builds `SAMPart3DDataset16Views`.  
- The dataset uses the renders/depth to associate 2D SAM masks with sampled 3D points, providing pseudo instance labels.  
- The tiny-cuda-nn instance/position MLPs in `pointcept/models/SAMPart3D.py` learn to cluster PTv3 features + positional cues into part-specific embeddings.  
- At the end of training, the script automatically runs inference over the saved renders, clusters embeddings with GPU HDBSCAN at every `val_scales_list` value, and writes both point-cloud and mesh segmentations.

## Visualization
- Colored `.ply` meshes in `exp/.../vis_pcd/last/` show the knight with each face tinted by its final part ID. You can inspect them directly in MeshLab/Blender.  
- `tools/highlight_parts.py` reprojects the mesh predictions back onto the multi-view renders, giving 2D overlays that highlight each part boundary in the best view for that part.
