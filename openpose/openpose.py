import os
import base64
import traceback
import tempfile
import subprocess
import json
from pathlib import Path
from typing import List

import numpy as np

# Map OpenPose COCO-18 (18 keypoints including Neck) to COCO-17 (standard format)
_IDX_MAP_18_TO_17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


def run_openpose_on_image(image_path, output_json_path, output_img_path=None):
    """Run the OpenPose binary on a single image (tries common flags).

    This mirrors the original implementation in `api_server.py`.
    Raises RuntimeError on failure.
    """
    openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
    num_gpu = os.environ.get('OPENPOSE_NUM_GPU', os.environ.get('NUM_GPU', '1'))
    num_gpu_start = os.environ.get('OPENPOSE_NUM_GPU_START', os.environ.get('NUM_GPU_START', '0'))
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    # Align defaults with local OpenPoseDemo usage
    net_res = os.environ.get('OPENPOSE_NET_RESOLUTION', '-1x368')
    render_thr = os.environ.get('OPENPOSE_RENDER_THRESHOLD', '0.2')

    base_args = [
        openpose_bin,
        "--model_folder", "/opt/openpose/models",
        "--write_json", str(output_json_path),
        "--display", "0",
        "--render_pose", "0",
        "--net_resolution", str(net_res),
        "--output_resolution", "-1x-1",
        "--number_people_max", "1",
        "--model_pose", "COCO",
        "--disable_blending", "false",
        "--scale_number", "1",
        "--scale_gap", "0.4",
        "--render_threshold", str(render_thr),
        "--num_gpu", str(num_gpu),
        "--num_gpu_start", str(num_gpu_start)
    ]

    if output_img_path and os.environ.get('OPENPOSE_WRITE_IMAGES', '0') == '1':
        base_args += ["--write_images", str(os.path.dirname(output_img_path)), "--render_pose", "1"]

    cmds_to_try = [base_args + ["--image_path", str(image_path)], base_args + ["--image_dir", str(os.path.dirname(image_path))]]
    last_err = None
    for cmd in cmds_to_try:
        env = os.environ.copy()
        if cuda_visible is not None:
            env['CUDA_VISIBLE_DEVICES'] = cuda_visible
        import time
        t0 = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        dt = time.time() - t0
        print(f"[DEBUG] OpenPose command took {dt:.3f}s: {' '.join(cmd)}")
        if result.returncode == 0:
            return
        last_err = result.stderr.decode('utf-8', errors='replace')

    err_msg = f"OpenPose failed after trying flags. Attempts:\n"
    for i, c in enumerate(cmds_to_try):
        err_msg += f"Attempt {i+1}: {' '.join(c)}\n"
    err_msg += f"Last stderr:\n{last_err}\n"
    raise RuntimeError(err_msg)


def run_openpose_on_dir(image_dir, output_json_path, output_img_path=None):
    """Run OpenPose on a directory of images using --image_dir."""
    openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
    net_res = os.environ.get('OPENPOSE_NET_RESOLUTION', '-1x368')
    render_thr = os.environ.get('OPENPOSE_RENDER_THRESHOLD', '0.2')
    num_gpu = os.environ.get('OPENPOSE_NUM_GPU', os.environ.get('NUM_GPU', '1'))
    num_gpu_start = os.environ.get('OPENPOSE_NUM_GPU_START', os.environ.get('NUM_GPU_START', '0'))

    # Clear any stale PyTorch CUDA cache before OpenPose execution
    try:
        import torch
        if torch.cuda.is_available():
            # Force PyTorch to initialize CUDA context on device 0
            try:
                torch.cuda.set_device(0)
                _ = torch.zeros(1).cuda()  # Force CUDA initialization
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[WARN] Could not initialize PyTorch CUDA device: {e}")
            torch.cuda.empty_cache()
            print(f"[DEBUG] PyTorch CUDA state: available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}, current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
    except Exception as e:
        print(f"[WARN] Could not clear PyTorch CUDA cache: {e}")

    cmd = [
        openpose_bin,
        "--image_dir", str(image_dir),
        "--model_folder", "/opt/openpose/models",
        "--write_json", str(output_json_path),
        "--display", "0",
        "--render_pose", "0",
        "--net_resolution", str(net_res),
        "--output_resolution", "-1x-1",
        "--number_people_max", "1",
        "--model_pose", "COCO",
        "--disable_blending", "false",
        "--scale_number", "1",
        "--scale_gap", "0.4",
        "--render_threshold", str(render_thr),
        "--num_gpu", str(num_gpu),
        "--num_gpu_start", str(num_gpu_start)
    ]
    # Only write images when explicitly enabled via env to match local
    if output_img_path and os.environ.get('OPENPOSE_WRITE_IMAGES', '0') == '1':
        cmd += ["--write_images", str(os.path.dirname(output_img_path))]
    
    # Ensure CUDA environment variables are set with correct priority
    env = os.environ.copy()
    env['CUDA_HOME'] = '/usr/local/cuda'
    # CRITICAL: Reconstruct LD_LIBRARY_PATH with CUDA libs first, excluding stubs
    cuda_libs = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/extras/CUPTI/lib64',
        '/usr/local/nvidia/lib',
        '/usr/local/nvidia/lib64'
    ]
    # Filter out stub libraries and OpenCV libs that might conflict
    existing_paths = [p for p in env.get('LD_LIBRARY_PATH', '').split(':') 
                      if p and 'stubs' not in p and 'cv2' not in p]
    # Put CUDA libs first, then other paths
    env['LD_LIBRARY_PATH'] = ':'.join(cuda_libs + existing_paths)
    # Unset CUDA_VISIBLE_DEVICES to let runtime handle it
    env.pop('CUDA_VISIBLE_DEVICES', None)
    
    import time
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    dt = time.time() - t0
    try:
        count = len(list(Path(image_dir).glob('*')))
    except Exception:
        count = 0
    print(f"[DEBUG] run_openpose_on_dir took {dt:.3f}s for {count} images")
    if result.returncode != 0:
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        # Enhanced diagnostics for CUDA detection failure (no retry with unsupported flags)
        if 'Cuda check failed' in stderr or 'no CUDA-capable device' in stderr:
            # Collect lightweight runtime diagnostics to aid operator
            diag = []
            try:
                import torch
                diag.append(f"torch.cuda.is_available={torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    diag.append(f"torch.cuda.device_count={torch.cuda.device_count()}")
                    diag.append(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
            except Exception as e:
                diag.append(f"torch_import_error={e}")
            # Environment variables relevant to GPU visibility
            for k in ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']:
                diag.append(f"env.{k}={os.environ.get(k, '')}")
            advice = (
                "OpenPose reports no CUDA device. Yet nvidia-smi may still work in base image. "
                "Possible causes: (1) Container not started with --gpus all; (2) CUDA_VISIBLE_DEVICES filters device; "
                "(3) Mismatch between CUDA version OpenPose was built with and host driver; (4) Running inside nested container/WSL without proper NVIDIA runtime. "
                "Actions: Confirm docker run uses --gpus all; remove manual NVIDIA_VISIBLE_DEVICES env overrides; try without setting CUDA_VISIBLE_DEVICES; verify driver >= required; test minimal CUDA program."
            )
            raise RuntimeError(
                "OpenPose CUDA device detection failed.\n" +
                f"Command: {' '.join(cmd)}\nSTDERR:\n{stderr}\n" +
                "Diagnostics:\n  - " + '\n  - '.join(diag) + '\n' +
                "Advice:\n  " + advice + '\n'
            )
        raise RuntimeError(
            f"OpenPose dir invocation failed. Command: {' '.join(cmd)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n"
        )


def _sanitize_person_list(person):
    for idx, kp in enumerate(person):
        try:
            x = kp[0] if kp[0] is not None else 0.0
            y = kp[1] if kp[1] is not None else 0.0
            c = kp[2] if kp[2] is not None else 0.0
            if not np.isfinite(x):
                x = 0.0
            if not np.isfinite(y):
                y = 0.0
            if not np.isfinite(c):
                c = 0.0
        except Exception:
            x, y, c = 0.0, 0.0, 0.0
        person[idx] = [float(x), float(y), float(c)]
    return person



def run_openpose_on_video(video_path, output_json_path, output_img_path=None):
    """Run OpenPose on a video file using --video."""
    openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
    net_res = os.environ.get('OPENPOSE_NET_RESOLUTION', '-1x368')
    render_thr = os.environ.get('OPENPOSE_RENDER_THRESHOLD', '0.2')

    cmd = [
        openpose_bin,
        "--video", str(video_path),
        "--model_folder", "/opt/openpose/models",
        "--write_json", str(output_json_path),
        "--display", "0",
        "--render_pose", "0",
        "--net_resolution", str(net_res),
        "--output_resolution", "-1x-1",
        "--number_people_max", "1",
        "--model_pose", "COCO",
        "--disable_blending", "false",
        "--scale_number", "1",
        "--scale_gap", "0.4",
        "--render_threshold", str(render_thr)
    ]
    if output_img_path and os.environ.get('OPENPOSE_WRITE_IMAGES', '0') == '1':
        cmd += ["--write_images", str(os.path.dirname(output_img_path))]
    env = os.environ.copy()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    import time
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    dt = time.time() - t0
    print(f"[DEBUG] run_openpose_on_video took {dt:.3f}s")
    if result.returncode != 0:
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        raise RuntimeError(f"OpenPose video invocation failed. Command: {' '.join(cmd)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n")


def OpenPoseProcessVideo(base64_video: str) -> List[List[List[List[float]]]]:
    """Process a base64-encoded MP4 video. Returns people-per-frame.

    Two-pass pipeline to mirror local preprocessing:
    1) Try --video. If fails, extract frames.
    2) First pass on raw frames to estimate a union person bbox (using keypoints).
    3) Crop frames to that bbox with padding and re-run OpenPose (higher-quality, fewer zeros).
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, 'input.mp4')
            with open(video_path, 'wb') as vf:
                vf.write(base64.b64decode(base64_video))

            output_json_dir = os.path.join(tmpdir, 'json')
            os.makedirs(output_json_dir, exist_ok=True)
            output_img_dir = os.path.join(tmpdir, 'img')
            os.makedirs(output_img_dir, exist_ok=True)

            frames_dir = None
            try:
                run_openpose_on_video(video_path, output_json_dir, output_img_dir)
            except Exception:
                traceback.print_exc()
                # Fallback: extract frames
                try:
                    import cv2
                    cap = cv2.VideoCapture(video_path)
                    frame_idx = 0
                    frames_dir = os.path.join(tmpdir, 'frames_raw')
                    os.makedirs(frames_dir, exist_ok=True)
                    success, frame = cap.read()
                    while success:
                        out_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(out_path, frame)
                        frame_idx += 1
                        success, frame = cap.read()
                    cap.release()
                except Exception:
                    traceback.print_exc()
                    return []

            # If we have JSON from video mode, skip to parsing; else run first pass on raw frames
            if not os.listdir(output_json_dir):
                if frames_dir is None:
                    return []
                # First pass: run OpenPose to get coarse keypoints
                run_openpose_on_dir(frames_dir, output_json_dir, output_img_dir)

            # Parse first-pass keypoints and estimate union bbox
            json_files = sorted([f for f in os.listdir(output_json_dir) if f.endswith('.json')])
            # If no outputs, bail
            if not json_files:
                return []

            # Collect bbox from ALL joints across frames (low confidence threshold for robust crop)
            xs, ys = [], []
            people_by_frame = []
            for jf in json_files:
                path = os.path.join(output_json_dir, jf)
                with open(path, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
                raw_people = jdata.get('people', [])
                ppl = []
                for p in raw_people:
                    if 'pose_keypoints_2d' in p:
                        kps = p['pose_keypoints_2d']
                        person_18 = [kps[idx:idx+3] for idx in range(0, len(kps), 3)]
                        if len(person_18) >= 18:
                            person = [person_18[op_idx] for op_idx in _IDX_MAP_18_TO_17]
                        else:
                            person = person_18
                        person = _sanitize_person_list(person)
                        ppl.append(person)
                people_by_frame.append(ppl)
                if ppl:
                    # use first person; collect ALL non-zero points (even low confidence)
                    for (x, y, c) in ppl[0]:
                        if x > 0.0 and y > 0.0:  # any detected point
                            xs.append(x)
                            ys.append(y)

            # If we cannot estimate bbox, just return first-pass results
            if not xs or not ys or len(xs) < 3:
                return people_by_frame

            # Compute union bbox with padding
            x_min = float(np.min(xs)); x_max = float(np.max(xs))
            y_min = float(np.min(ys)); y_max = float(np.max(ys))
            w = x_max - x_min; h = y_max - y_min
            pad_x = float(os.environ.get('CROP_PAD_X', '0.10'))
            pad_y = float(os.environ.get('CROP_PAD_Y', '0.10'))
            x1 = max(0, int(x_min - w * pad_x))
            y1 = max(0, int(y_min - h * pad_y))
            x2 = int(x_max + w * pad_x)
            y2 = int(y_max + h * pad_y)

            # Crop frames and re-run OpenPose for higher-quality detection
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame_w - 1, x2); y2 = min(frame_h - 1, y2)
            crop_w = max(1, x2 - x1); crop_h = max(1, y2 - y1)
            cropped_dir = os.path.join(tmpdir, 'frames_cropped')
            os.makedirs(cropped_dir, exist_ok=True)
            idx = 0
            ok, frame = cap.read()
            while ok:
                crop = frame[y1:y2, x1:x2]
                out_path = os.path.join(cropped_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(out_path, crop)
                idx += 1
                ok, frame = cap.read()
            cap.release()

            # Second pass: run OpenPose on cropped frames (net_resolution can be increased via env)
            output_json_dir_2 = os.path.join(tmpdir, 'json2')
            os.makedirs(output_json_dir_2, exist_ok=True)
            run_openpose_on_dir(cropped_dir, output_json_dir_2, None)

            # Parse second-pass results
            json_files_2 = sorted([f for f in os.listdir(output_json_dir_2) if f.endswith('.json')])
            result_by_frame = []
            for jf in json_files_2:
                path = os.path.join(output_json_dir_2, jf)
                with open(path, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
                raw_people = jdata.get('people', [])
                ppl = []
                for p in raw_people:
                    if 'pose_keypoints_2d' in p:
                        kps = p['pose_keypoints_2d']
                        person_18 = [kps[idx:idx+3] for idx in range(0, len(kps), 3)]
                        if len(person_18) >= 18:
                            person = [person_18[op_idx] for op_idx in _IDX_MAP_18_TO_17]
                        else:
                            person = person_18
                        person = _sanitize_person_list(person)
                        ppl.append(person)
                result_by_frame.append(ppl)
            return result_by_frame
    except Exception:
        traceback.print_exc()
        return []
