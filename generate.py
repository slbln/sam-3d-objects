import sys
from pathlib import Path

sys.path.append("notebook")
from inference import Inference, load_image, load_mask

tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

pairs = [
    ("inputs/img1.png", "inputs/img1_mask.png", "outputs/object1"),
    ("inputs/img2.png", "inputs/img2_mask.png", "outputs/object2"),
    ("inputs/img3.png", "inputs/img3_mask.png", "outputs/object3"),
]

for image_path, mask_path, out_dir in pairs:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    mask = load_mask(mask_path)

    rgba = inference.merge_mask_to_rgba(image, mask)

    output = inference._pipeline.run(
        rgba,
        None,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=False,
        use_vertex_color=False,
        stage1_inference_steps=None,
        pointmap=None,
    )

    # 高斯结果（备份）
    output["gs"].save_ply(str(out_dir / "splat.ply"))

    # 中间结果：带材质/贴图信息的 GLB
    if output["glb"] is not None:
        output["glb"].export(str(out_dir / "model.glb"))

    print(f"done: {out_dir}")