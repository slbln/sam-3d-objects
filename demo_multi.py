# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys
import gc
import json
import time
import math
import traceback
import subprocess
from pathlib import Path

# -----------------------------
# Offline mode for HF libs
# -----------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import numpy as np
from PIL import Image

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_masks

# =========================================================
# Global options
# =========================================================
SKIP_EXISTING_OUTPUTS = True  # If True, skip generation when GLB and pose JSON already exist


# =========================================================
# Utils
# =========================================================
def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now_str()}] {msg}", flush=True)


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "cuda error: out of memory" in msg
        or "cuda failed with error 2" in msg
    )


def is_nameerror_gaussian(exc: Exception) -> bool:
    msg = str(exc)
    return "GaussianRasterizationSettings" in msg or "GaussianRasterizer" in msg


def to_pil(img):
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(img)}")


def from_pil_like(original, pil_img):
    if isinstance(original, Image.Image):
        return pil_img
    if isinstance(original, np.ndarray):
        return np.array(pil_img)
    return pil_img


def resize_like(x, scale, is_mask=False):
    if abs(scale - 1.0) < 1e-6:
        return x
    pil = to_pil(x)
    w, h = pil.size
    new_w = max(64, int(round(w * scale)))
    new_h = max(64, int(round(h * scale)))
    resample = Image.NEAREST if is_mask else Image.LANCZOS
    pil_resized = pil.resize((new_w, new_h), resample=resample)
    return from_pil_like(x, pil_resized)


def mask_area(mask):
    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return int((arr > 0).sum())


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def apply_pose_to_mesh(mesh, rotation, translation, scale):
    """
    Apply pose transformation to mesh vertices.

    The GLB vertices are in z-up format. The pose operates in y-up format.
    We convert z-up to y-up (same as get_mesh), then apply the pose using the
    same convention as compose_transform / SceneVisualizer.object_pointcloud:
        v_world = (v_yup * scale) @ R + translation
    where R = quaternion_to_matrix(rotation).
    """
    import numpy as np

    verts = np.asarray(mesh.vertices).copy().astype(np.float32)

    # B: z-up to y-up (same matrix as in get_mesh from layout_post_optimization_utils.py)
    B = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ], dtype=np.float32)

    # Convert z-up to y-up
    verts_yup = verts @ B.T

    # quaternion to rotation matrix (w, x, y, z) — standard formula matching pytorch3d
    quat = np.array(rotation).reshape(-1).astype(np.float32)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float32)

    s = np.array(scale).reshape(-1).astype(np.float32)
    t = np.array(translation).reshape(-1).astype(np.float32)

    # Apply pose: v_world = (v_yup * scale) @ R + translation
    # This matches compose_transform(scale, R, translation).transform_points(v)
    # used in SceneVisualizer.object_pointcloud and make_scene.
    verts_world = (verts_yup * s) @ R + t

    mesh.vertices = verts_world
    return mesh


# =========================================================
# GPU helpers
# =========================================================
def query_gpus():
    """
    Return list of dicts:
    [
      {
        'index': 0,
        'name': 'NVIDIA GeForce RTX 4090',
        'memory.total': 24564,
        'memory.used': 1200,
        'memory.free': 23364
      },
      ...
    ]
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        gpus = []
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory.total": int(parts[2]),
                    "memory.used": int(parts[3]),
                    "memory.free": int(parts[4]),
                }
            )
        return gpus
    except Exception as e:
        log(f"Warning: failed to query nvidia-smi: {e}")
        return []


def pick_best_gpu(min_free_mb=8000):
    """
    Pick the GPU with the most free memory.
    """
    gpus = query_gpus()
    if not gpus:
        return 0

    # 先按空闲显存降序
    gpus = sorted(gpus, key=lambda x: x["memory.free"], reverse=True)

    # 优先选空闲显存足够的
    for g in gpus:
        if g["memory.free"] >= min_free_mb:
            return g["index"]

    # 否则退而求其次：选最空的
    return gpus[0]["index"]


def set_cuda_device(device_index: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.cuda.set_device(device_index)
    log(f"Using GPU {device_index}: {torch.cuda.get_device_name(device_index)}")


# =========================================================
# Backend helpers
# =========================================================
def can_import_inria_backend():
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings  # noqa: F401
        from diff_gaussian_rasterization import GaussianRasterizer  # noqa: F401
        return True
    except Exception:
        return False


def set_backend_safely(inference_obj, preferred="auto"):
    """
    优先：
    - 如果 inria backend 可用，就用 inria
    - 否则回退 gsplat
    """
    backend = None

    if preferred == "inria":
        backend = "inria" if can_import_inria_backend() else "gsplat"
    elif preferred == "gsplat":
        backend = "gsplat"
    else:
        backend = "inria" if can_import_inria_backend() else "gsplat"

    # 尽量兼容不同版本的 pipeline / renderer 挂载位置
    touched = False

    # 常见位置 1
    if hasattr(inference_obj, "_pipeline"):
        p = inference_obj._pipeline

        if hasattr(p, "rendering_options") and isinstance(p.rendering_options, dict):
            p.rendering_options["backend"] = backend
            touched = True
        elif hasattr(p, "rendering_options"):
            try:
                p.rendering_options["backend"] = backend
                touched = True
            except Exception:
                pass

        # 有些版本 renderer 在 _renderer 上
        if hasattr(p, "_renderer") and hasattr(p._renderer, "rendering_options"):
            try:
                p._renderer.rendering_options["backend"] = backend
                touched = True
            except Exception:
                pass

        # 有些版本 renderer 在 renderer 上
        if hasattr(p, "renderer") and hasattr(p.renderer, "rendering_options"):
            try:
                p.renderer.rendering_options["backend"] = backend
                touched = True
            except Exception:
                pass

    # 尝试从 inference 本体设置
    if hasattr(inference_obj, "rendering_options"):
        try:
            inference_obj.rendering_options["backend"] = backend
            touched = True
        except Exception:
            pass

    if not touched:
        log("Warning: backend option may not have been injected successfully.")

    log(f"Rendering backend = {backend}")
    return backend


# =========================================================
# Per-object robust run
# =========================================================
def run_single_object(
    inference,
    image,
    mask,
    object_idx,
    output_dir,
    retry_scales,
    backend_mode="auto",
    min_mask_area=64,
):
    """
    Returns a dict with result info.
    """
    info = {
        "object_index": object_idx,
        "success": False,
        "backend": None,
        "scale": None,
        "gpu": None,
        "mask_area": None,
        "glb_path": None,
        "ply_path": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }

    area = mask_area(mask)
    info["mask_area"] = area

    glb_path = os.path.join(output_dir, f"model_{object_idx:03d}.glb")
    pose_path = os.path.join(output_dir, f"model_{object_idx:03d}_pose.json")

    # Skip if outputs already exist and SKIP_EXISTING_OUTPUTS is enabled
    if SKIP_EXISTING_OUTPUTS and os.path.exists(glb_path) and os.path.exists(pose_path):
        info["glb_path"] = glb_path
        info["pose_path"] = pose_path
        info["success"] = True
        log(f"Object {object_idx:03d}: skipped (outputs exist)")
        return info

    if area < min_mask_area:
        info["error_type"] = "tiny_mask"
        info["error_message"] = f"mask area too small: {area}"
        log(f"Object {object_idx:03d}: skipped tiny mask (area={area})")
        return info

    for scale in retry_scales:
        try:
            clear_vram()

            # 设置 backend
            backend = set_backend_safely(inference, preferred=backend_mode)
            info["backend"] = backend
            info["scale"] = scale

            current_gpu = torch.cuda.current_device()
            info["gpu"] = current_gpu

            log(
                f"Object {object_idx:03d}: try scale={scale:.2f}, "
                f"mask_area={area}, gpu={current_gpu}, backend={backend}"
            )

            image_try = resize_like(image, scale, is_mask=False)
            mask_try = resize_like(mask, scale, is_mask=True)

            rgba = inference.merge_mask_to_rgba(image_try, mask_try)

            with torch.inference_mode():
                output = inference._pipeline.run(
                    rgba,
                    None,  # mask already merged
                    seed=42,
                    stage1_only=False,
                    with_mesh_postprocess=False,   # 保纹理，先关这个省显存
                    with_texture_baking=True,      # 你需要带纹理 GLB
                    with_layout_postprocess=True,
                    use_vertex_color=False,
                    stage1_inference_steps=None,
                    pointmap=None,
                )

            # 保存输出
            # ply_path = os.path.join(output_dir, f"splat_{object_idx:03d}.ply")
            glb_path = os.path.join(output_dir, f"model_{object_idx:03d}.glb")

            # if output.get("gs", None) is not None:
            #     output["gs"].save_ply(ply_path)
            #     info["ply_path"] = ply_path

            if output.get("glb", None) is not None:
                output["glb"].export(glb_path)
                info["glb_path"] = glb_path

                # Save pose metadata: rotation, translation, scale, center
                verts = np.asarray(output["glb"].vertices)
                vmin, vmax = verts.min(axis=0), verts.max(axis=0)
                center = ((vmax + vmin) / 2.0).tolist()

                pose_info = {
                    "object_index": object_idx,
                    "glb_path": glb_path,
                    "rotation": output["rotation"].detach().cpu().tolist(),
                    "translation": output["translation"].detach().cpu().tolist(),
                    "scale": output["scale"].detach().cpu().tolist(),
                    "center": center,
                }
                pose_path = os.path.join(output_dir, f"model_{object_idx:03d}_pose.json")
                with open(pose_path, "w", encoding="utf-8") as f:
                    json.dump(pose_info, f, ensure_ascii=False, indent=2)
                info["pose_path"] = pose_path

                info["success"] = True
                log(f"Object {object_idx:03d}: success -> {glb_path}")
                return info

            # 没抛异常但也没 glb，继续降分辨率试
            log(f"Object {object_idx:03d}: no GLB at scale={scale:.2f}, retry smaller")

        except RuntimeError as e:
            if is_oom_error(e):
                log(f"Object {object_idx:03d}: OOM at scale={scale:.2f}, retry smaller")
                info["error_type"] = "cuda_oom"
                info["error_message"] = str(e)
                info["traceback"] = traceback.format_exc()
                clear_vram()
                continue
            else:
                # 某些 backend/runtime 异常，尝试切 backend 再下一轮
                log(f"Object {object_idx:03d}: RuntimeError at scale={scale:.2f}: {e}")
                info["error_type"] = "runtime_error"
                info["error_message"] = str(e)
                info["traceback"] = traceback.format_exc()
                clear_vram()
                continue

        except NameError as e:
            # 专门针对 GaussianRasterizationSettings 这类错误
            if is_nameerror_gaussian(e):
                log(
                    f"Object {object_idx:03d}: inria backend unavailable at "
                    f"scale={scale:.2f}, fallback and retry"
                )
                info["error_type"] = "backend_nameerror"
                info["error_message"] = str(e)
                info["traceback"] = traceback.format_exc()

                # 下次循环会再次 set_backend_safely(auto)，通常会落到 gsplat
                clear_vram()
                continue
            else:
                log(f"Object {object_idx:03d}: NameError at scale={scale:.2f}: {e}")
                info["error_type"] = "name_error"
                info["error_message"] = str(e)
                info["traceback"] = traceback.format_exc()
                clear_vram()
                continue

        except Exception as e:
            log(f"Object {object_idx:03d}: Exception at scale={scale:.2f}: {e}")
            info["error_type"] = type(e).__name__
            info["error_message"] = str(e)
            info["traceback"] = traceback.format_exc()
            clear_vram()
            continue

        finally:
            clear_vram()

    log(f"Object {object_idx:03d}: failed after all retries")
    return info


# =========================================================
# Main
# =========================================================
def main():
    # --------------- config ---------------
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"

    image_path = "notebook/images/chair_001/image.png"
    output_dir = "outputs/chair_001"

    # 从高到低自动重试
    retry_scales = [1.00, 0.85, 0.70, 0.55, 0.45]

    # auto: 有 inria 就 inria，没有就 gsplat
    # 如果你现在环境没装 diff_gaussian_rasterization，auto 会优先回退 gsplat
    backend_mode = "auto"

    # 特别小的 mask 没必要跑
    min_mask_area = 64

    # --------------- init ---------------
    ensure_dir(output_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU.")

    gpus = query_gpus()
    if gpus:
        log("Detected GPUs:")
        for g in gpus:
            log(
                f"  GPU {g['index']}: {g['name']} | "
                f"used {g['memory.used']} MiB / total {g['memory.total']} MiB | "
                f"free {g['memory.free']} MiB"
            )

    best_gpu = pick_best_gpu(min_free_mb=7000)
    set_cuda_device(best_gpu)

    log(f"Loading inference from {config_path}")
    inference = Inference(config_path, compile=False)

    # 提前设置一次 backend
    set_backend_safely(inference, preferred=backend_mode)

    image = load_image(image_path)
    masks = load_masks(os.path.dirname(image_path))
    log(f"Found {len(masks)} masks")

    # --------------- process all ---------------
    summary = {
        "started_at": now_str(),
        "config_path": config_path,
        "image_path": image_path,
        "output_dir": output_dir,
        "retry_scales": retry_scales,
        "backend_mode": backend_mode,
        "results": [],
    }

    success_count = 0
    fail_count = 0

    for i, mask in enumerate(masks):
        log("=" * 80)
        log(f"Processing object {i + 1}/{len(masks)}")

        result = run_single_object(
            inference=inference,
            image=image,
            mask=mask,
            object_idx=i,
            output_dir=output_dir,
            retry_scales=retry_scales,
            backend_mode=backend_mode,
            min_mask_area=min_mask_area,
        )

        summary["results"].append(result)

        if result["success"]:
            success_count += 1
        else:
            fail_count += 1

        # 每个 object 之间再清一次
        clear_vram()

        # 实时写日志，避免中途断掉什么都没留下
        with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["finished_at"] = now_str()
    summary["success_count"] = success_count
    summary["fail_count"] = fail_count

    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 额外写一份简洁失败清单
    failed_items = [
        {
            "object_index": r["object_index"],
            "mask_area": r["mask_area"],
            "gpu": r["gpu"],
            "backend": r["backend"],
            "last_scale": r["scale"],
            "error_type": r["error_type"],
            "error_message": r["error_message"],
        }
        for r in summary["results"]
        if not r["success"]
    ]
    with open(os.path.join(output_dir, "failed_objects.json"), "w", encoding="utf-8") as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)

    # =========================================================
    # Combine successful models into a single scene
    # =========================================================
    import trimesh

    log("=" * 80)
    log("Combining successful models into a scene...")

    scene = trimesh.Scene()
    combined_count = 0

    for result in summary["results"]:
        if result["success"] and result.get("glb_path"):
            glb_path = result["glb_path"]
            pose_path = result.get("pose_path")
            if os.path.exists(glb_path) and pose_path and os.path.exists(pose_path):
                try:
                    with open(pose_path, "r", encoding="utf-8") as f:
                        pose = json.load(f)

                    obj_mesh = trimesh.load(glb_path)
                    # Handle scene files with multiple meshes
                    if isinstance(obj_mesh, trimesh.Scene):
                        for name, geom in obj_mesh.geometry.items():
                            # apply pose transform to each sub-geometry
                            transformed_geom = apply_pose_to_mesh(
                                geom,
                                pose["rotation"],
                                pose["translation"],
                                pose["scale"],
                            )
                            scene.add_geometry(transformed_geom, node_name=f"object_{result['object_index']:03d}_{name}")
                    else:
                        transformed_mesh = apply_pose_to_mesh(
                            obj_mesh,
                            pose["rotation"],
                            pose["translation"],
                            pose["scale"],
                        )
                        scene.add_geometry(transformed_mesh, node_name=f"object_{result['object_index']:03d}")
                    combined_count += 1
                    log(f"  Added: {glb_path} with pose transform")
                except Exception as e:
                    log(f"  Failed to load {glb_path}: {e}")

    if combined_count > 0:
        scene_glb_path = os.path.join(output_dir, "scene.glb")
        scene.export(scene_glb_path)
        log(f"Scene exported to: {scene_glb_path} ({combined_count} objects)")
    else:
        log("No objects to combine into scene.")

    log("=" * 80)
    log(f"Done. success={success_count}, failed={fail_count}")
    log(f"Summary written to: {os.path.join(output_dir, 'run_summary.json')}")
    log(f"Failed list written to: {os.path.join(output_dir, 'failed_objects.json')}")


if __name__ == "__main__":
    main()