"""
Controller (OpenPose + Metrics Orchestrator)
===========================================

이 모듈은 다음을 수행합니다.
- 입력(2D: mp4, 3D: zip)을 S3에서 다운로드하여 /tmp 하위에서 처리
- OpenPose로 포즈 키포인트 추출 → 프레임별 JSON/이미지 생성 → DataFrame 구성(df_2d/df_3d)
- 메트릭 러너(metric_algorithm)를 호출해 각 메트릭을 실행하고 산출물(csv/mp4/summary)을 수집
- 최종 결과 JSON(<job_id>.json)을 저장하고, 산출물 디렉터리(csv/, mp4/, img/, openpose_img/)를 정리

환경 변수
- 입력 버킷: S3_VIDEO_BUCKET_NAME 또는 S3_BUCKET/AWS_S3_BUCKET
- 결과 버킷: S3_RESULT_BUCKET_NAME 또는 RESULT_S3_BUCKET

주의
- 본 파일은 openpose 패키지 경로가 PYTHONPATH에 잡혀 있어야 합니다
    (예: .../golf/containers/skeleton_metric-api 가 sys.path에 포함되어야 openpose 임포트 가능).
"""

import traceback
import json
import os
import boto3
from pathlib import Path
from typing import Optional, Tuple, List
import tempfile
import zipfile
import shutil

import numpy as np
import pandas as pd
# Import mmaction_client robustly: when this module is executed as a top-level script
# (e.g. inside a container via `python controller.py` or uvicorn importing module by name),
# relative imports like `from . import mmaction_client` can fail with
# "attempted relative import with no known parent package". Try relative first,
# then fall back to absolute import which works when the package root is on sys.path.
try:
    from . import mmaction_client
except Exception:
    import mmaction_client

from openpose.skeleton_interpolate import interpolate_sequence
from openpose.openpose import run_openpose_on_video, run_openpose_on_dir, _sanitize_person_list
from metric_algorithm.smoothing import (
    calculate_z_bounds, validate_z_value,
    filter_z_outliers_by_frame_delta,
    filter_z_outliers_by_frame_consistency
)



# ============================================================================
# YOLO Person Detection for Cropping
# ============================================================================

def _load_yolo_detector():
    """Load YOLO model for person detection (matches local video_crop.py approach).
    
    Returns a detect function that takes a BGR numpy array and returns
    list of (xyxy, conf, cls) tuples for detected objects.
    """
    try:
        # Import torch first to ensure CUDA is properly initialized
        import torch
        # Force CUDA initialization to detect devices properly
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0)
                _ = torch.zeros(1).cuda()  # Force initialization
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[WARN] YOLO: Could not initialize CUDA device: {e}")
            torch.cuda.empty_cache()
        
        from ultralytics import YOLO
        # Use pre-copied model file to avoid runtime download
        model_path = Path('/opt/skeleton_metric-api/yolov8n.pt')
        if not model_path.exists():
            print(f"[WARN] YOLO model not found at {model_path}, trying default download")
            model = YOLO('yolov8n.pt')
        else:
            model = YOLO(str(model_path))
        
        # Force CPU mode to avoid CUDA conflicts with OpenPose
        # YOLO detection is fast enough on CPU for cropping purposes
        model.to('cpu')
        
        def detect(img, device='cpu'):
            # Always use CPU for YOLO to avoid CUDA conflicts
            results = model.predict(img, imgsz=640, device='cpu', verbose=False)
            out = []
            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy())
                    cls = int(b.cls[0].cpu().numpy())
                    out.append((xyxy, conf, cls))
            return out
        
        # Clear CUDA cache after YOLO initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return detect
    except Exception as e:
        print(f"[WARN] Failed to load YOLO: {e}")
        print(traceback.format_exc())
        return None


def _detect_person_bbox_from_video(video_path: Path, detect_fn, sample_fraction: float = 0.25):
    """Detect person bbox from video using YOLO on sampled frames.
    
    Returns union bbox (x1, y1, x2, y2) in frame coordinates, or None if no person detected.
    Matches local video_crop.py sampling and clustering logic.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample frames uniformly
    target = max(1, int(frame_count * sample_fraction))
    import numpy as np
    indices = np.linspace(0, frame_count - 1, num=min(target, frame_count), dtype=int)
    indices = np.unique(indices).tolist()
    
    all_boxes = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = detect_fn(frame)
        # Filter for person class (class 0 in COCO)
        for (xyxy, conf, cls) in dets:
            if cls == 0:  # person
                all_boxes.append(xyxy)
    cap.release()
    
    if not all_boxes:
        return None
    
    # Compute union bbox
    arr = np.array(all_boxes)
    x1 = float(arr[:, 0].min())
    y1 = float(arr[:, 1].min())
    x2 = float(arr[:, 2].max())
    y2 = float(arr[:, 3].max())
    
    return (x1, y1, x2, y2)


# ============================================================================
# OpenPose + Metrics Orchestration
# ============================================================================


def _convert_nan_to_none(obj):
    """Recursively convert NaN, Inf, -Inf to None (null in JSON).
    
    프론트엔드가 JSON의 NaN을 처리하지 못하므로, 
    모든 NaN/Inf를 None으로 변환하여 null로 직렬화되도록 함.
    numpy scalar도 처리.
    """
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        # numpy scalar 타입
        try:
            val = float(obj)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        except Exception:
            return None
    elif isinstance(obj, dict):
        return {k: _convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_nan_to_none(item) for item in obj)
    return obj


def _safe_write_json(path: Path, obj: dict):
    """Atomically write JSON to `path` by writing to a temp file and replacing.

    Uses custom JSON encoder that converts NaN/Inf to null.
    """
    try:
        # Convert all NaN/Inf to None before serialization
        obj_clean = _convert_nan_to_none(obj)
        
        tmp = Path(str(path) + '.tmp')
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(obj_clean, f, ensure_ascii=False, indent=2, default=str)
        # atomic replace
        try:
            tmp.replace(path)
        except Exception:
            # fallback to os.replace
            os.replace(str(tmp), str(path))
    except Exception:
        # best-effort: try non-atomic write
        try:
            obj_clean = _convert_nan_to_none(obj)
            with path.open('w', encoding='utf-8') as f:
                json.dump(obj_clean, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass


def _extract_mmaction_result(wrapper_obj):
    """Extract the MMACTION parsed result from the POST wrapper JSON.

    The client worker writes a wrapper like:
      { 'ok': True, 'status_code': 200, 'resp_json': <server-json> }

    <server-json> may be either the direct parsed result dict, or a dict
    containing a 'result' key. This helper returns the most-likely parsed
    result dict so controller can attach it under `stgcn_inference`.
    """
    try:
        if not isinstance(wrapper_obj, dict):
            return wrapper_obj
        # Prefer explicit resp_json entry
        inner = wrapper_obj.get('resp_json') if 'resp_json' in wrapper_obj else wrapper_obj.get('resp') if 'resp' in wrapper_obj else None
        if isinstance(inner, dict):
            # prefer an explicit 'parsed' field if present
            if 'parsed' in inner and isinstance(inner['parsed'], dict):
                return inner['parsed']
            if 'result' in inner and isinstance(inner['result'], dict):
                return inner['result']
            return inner
        # fallback: top-level 'parsed' or 'result'
        if 'parsed' in wrapper_obj and isinstance(wrapper_obj['parsed'], dict):
            return wrapper_obj['parsed']
        if 'result' in wrapper_obj and isinstance(wrapper_obj['result'], dict):
            return wrapper_obj['result']
        return wrapper_obj
    except Exception:
        return wrapper_obj


def _merge_stgcn_resp(resp_path: Path, response_payload: dict):
    """Read a response JSON file and merge the extracted MMACTION parsed result
    into `response_payload['stgcn_inference']`.
    """
    try:
        txt = Path(resp_path).read_text(encoding='utf-8')
        obj = json.loads(txt)
    except Exception:
        return
    try:
        response_payload['stgcn_inference'] = _extract_mmaction_result(obj)
    except Exception:
        response_payload['stgcn_inference'] = obj


def run_metrics_in_process(dimension: str, ctx: dict):
    """메트릭 실행 훅.

    Parameters
    ----------
    dimension : str
        '2d' 또는 '3d'. 결과 JSON이나 메트릭 내부 로직에서 필요.
    ctx : dict
        `process_and_save` 함수의 locals()를 기반으로 한 문맥. 여기에는
        df_2d / df_3d (존재 시), people_sequence, dest_dir, job_id 등이 포함.

    Behavior
    --------
    - `metric_algorithm.run_metrics_from_context` 를 찾아 실행합니다.
    - 메트릭 모듈/실행 실패는 전체 파이프라인을 멈추지 않고 None 반환.

    Returns
    -------
    dict | None
        메트릭 실행 결과(모듈별 딕셔너리) 또는 실패 시 None.
    """
    try:
        try:
            from metric_algorithm import run_metrics_from_context
        except Exception:
            traceback.print_exc()
            return None

        dest_dir = ctx.get('dest_dir') or ctx.get('dest') or None
        job_id = ctx.get('job_id') or ctx.get('job') or None

        if dest_dir is None or job_id is None:
            try:
                res = run_metrics_from_context(
                    ctx,
                    dest_dir=str(dest_dir) if dest_dir is not None else '.',
                    job_id=str(job_id) if job_id is not None else 'unknown',
                    dimension=dimension,
                )
            except Exception:
                traceback.print_exc()
                return None
        else:
            try:
                res = run_metrics_from_context(
                    ctx,
                    dest_dir=str(dest_dir),
                    job_id=str(job_id),
                    dimension=dimension,
                )
            except Exception:
                traceback.print_exc()
                return None
        return res
    except Exception:
        traceback.print_exc()
        return None


def process_and_save(s3_key: str, dimension: str, job_id: str, turbo_without_skeleton: bool, dest_dir: Path, is_local: bool = False):
    """입력 다운로드 → OpenPose 실행 → 보간 → 결과 저장(+메트릭 실행)까지 전체 파이프라인.

    매개변수
    - s3_key: 입력 객체 키(2D: mp4, 3D: zip) 또는 로컬 파일 경로
    - dimension: '2d' | '3d'
    - job_id: 산출물 파일명 접두에 쓰일 식별자
    - turbo_without_skeleton: (예약) skeleton 추출 스킵 플래그. 현재 로직에서는 사용하지 않음.
    - dest_dir: 산출물 저장 디렉터리 경로
    - is_local: True이면 s3_key 대신 dest_dir/local_input.* 파일 사용

    처리 개요
    1) is_local이 False: S3에서 입력 다운로드 (/tmp 하위 작업 디렉터리 사용)
       is_local이 True: dest_dir/local_input.* 파일을 사용
    2) 2D: mp4 → run_openpose_on_video → 프레임별 JSON 파싱 → df_2d 생성 → 원본 프레임 추출(img/)
       3D: zip → color/, depth/ 추출 → run_openpose_on_dir(color/) → 깊이 샘플링으로 Z 추정 → intrinsics로 X/Y 계산 → df_3d 생성
    3) interpolate_sequence로 people_sequence 생성(프레임별 첫 번째 사람 기준)
    4) <job_id>.json 저장 + OpenPose 렌더 이미지 openpose_img/ 보존
    5) run_metrics_in_process로 메트릭 실행 → 결과(csv, overlay mp4, summary 등)를 job json에 병합
    6) 산출물 정리(csv/, mp4/ 하위로 이동) 후 job json 경로 업데이트

    반환
    - response_payload(dict): message, people_sequence, frame_count, skeleton_rows/columns 등 포함
    """
    try:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Initialize fps variable for metrics pipeline (will be set from intrinsics if available)
        fps = None

        # 임시 작업 디렉터리 생성
        tmpdir = tempfile.mkdtemp(dir='/tmp')
        tmp_dir = Path(tmpdir)
        output_json_dir = tmp_dir / 'json'
        output_json_dir.mkdir(parents=True, exist_ok=True)
        output_img_dir = tmp_dir / 'img'
        output_img_dir.mkdir(parents=True, exist_ok=True)

        # S3 클라이언트 초기화 (로컬 모드가 아닐 때만)
        s3 = None
        bucket = None
        key = None
        
        if not is_local:
            # 입력 버킷을 환경변수에서 결정합니다.
            bucket = os.environ.get('S3_VIDEO_BUCKET_NAME') or os.environ.get('S3_BUCKET') or os.environ.get('AWS_S3_BUCKET')
            if not bucket:
                raise RuntimeError('S3_VIDEO_BUCKET_NAME (or S3_BUCKET/AWS_S3_BUCKET) environment variable is not set')
            s3 = boto3.client('s3')
            key = s3_key.lstrip('/')

        result_by_frame = []

        if dimension == '2d':  # --- 2D 처리 경로 ---
            # download mp4 (로컬 또는 S3)
            local_video = tmp_dir / 'input.mp4'
            
            if is_local:
                print(f"[DEBUG] 로컬 MP4 파일 복사 중...")
                local_input_mp4 = dest_dir / 'local_input.mp4'
                if not local_input_mp4.exists():
                    raise RuntimeError(f"로컬 파일 모드인데 local_input.mp4를 찾을 수 없습니다: {local_input_mp4}")
                import shutil
                shutil.copy(str(local_input_mp4), str(local_video))
                print(f"[DEBUG] 복사 완료: {local_video}")
            else:
                print(f"[DEBUG] S3에서 MP4 다운로드 중: {bucket}/{key}")
                s3.download_file(bucket, key, str(local_video))
                print(f"[DEBUG] 다운로드 완료: {local_video}")
                
                # Also save a copy to dest_dir/local_input.mp4 for debugging/inspection
                try:
                    import shutil
                    local_input_mp4 = dest_dir / 'local_input.mp4'
                    shutil.copy2(str(local_video), str(local_input_mp4))
                    print(f"[DEBUG] S3 파일 복사본 저장: {local_input_mp4}")
                except Exception as e:
                    print(f"[WARN] S3 파일 복사본 저장 실패: {e}")

            import cv2
            cap = cv2.VideoCapture(str(local_video))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Extract fps from video metadata for metrics pipeline
            fps_float = cap.get(cv2.CAP_PROP_FPS)
            fps = int(fps_float) if fps_float > 0 else None
            cap.release()
            print(f"[INFO] Video dimensions: {frame_width}x{frame_height}")
            print(f"[DEBUG] 2D path: fps_float={fps_float}, fps={fps}")
            
            # Ensure fps has a valid value for metrics pipeline
            if fps is None or fps <= 0:
                fps = 60  # Default to 60fps if not available from metadata
                print(f"[INFO] fps not available in video metadata, using default={fps}")
            else:
                print(f"[INFO] Extracted fps={fps} from 2D video")

            crop_strategy = os.environ.get('OPENPOSE_CROP_STRATEGY', 'yolo').lower().strip()
            if crop_strategy not in ('none', 'yolo', 'two_pass_keypoints'):
                print(f"[WARN] Unknown OPENPOSE_CROP_STRATEGY={crop_strategy}, falling back to 'yolo'")
                crop_strategy = 'yolo'
            debug_crop_info = {'strategy': crop_strategy}

            pad_x = float(os.environ.get('CROP_PAD_X', '0.15'))
            pad_y = float(os.environ.get('CROP_PAD_Y', '0.15'))

            # Helper: extract all frames to a directory
            def _extract_frames(src_video: Path, out_dir: Path):
                out_dir.mkdir(parents=True, exist_ok=True)
                c = cv2.VideoCapture(str(src_video))
                i = 0
                ok, fr = c.read()
                while ok:
                    cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), fr)
                    i += 1
                    ok, fr = c.read()
                c.release()
                return i

            # Strategy implementations
            if crop_strategy == 'none':
                print("[INFO] Using 'none' crop strategy: full-frame OpenPose")
                crop_bbox = (0, 0, frame_width, frame_height)
                frames_dir = tmp_dir / 'frames_full'
                count = _extract_frames(local_video, frames_dir)
                print(f"[INFO] Extracted {count} frames (full) for first/only pass")
                run_openpose_on_dir(str(frames_dir), str(output_json_dir), str(output_img_dir / 'out'))
                debug_crop_info['pass1_frame_count'] = count
                debug_crop_info['int_bbox'] = list(crop_bbox)
                debug_crop_info['float_bbox'] = [0.0, 0.0, float(frame_width), float(frame_height)]

            elif crop_strategy == 'yolo':
                print("[INFO] Using 'yolo' crop strategy")
                yolo_detect = _load_yolo_detector()
                person_bbox = None
                if yolo_detect is not None:
                    try:
                        person_bbox = _detect_person_bbox_from_video(local_video, yolo_detect, sample_fraction=0.25)
                        if person_bbox:
                            print(f"[INFO] YOLO detected person bbox: {person_bbox}")
                    except Exception as e:
                        print(f"[WARN] YOLO detection failed: {e}")

                if person_bbox is not None:
                    x1f, y1f, x2f, y2f = person_bbox
                    w, h = x2f - x1f, y2f - y1f
                    # keep float for debug; use int for actual crop indices
                    fx1 = max(0.0, x1f - w * pad_x)
                    fy1 = max(0.0, y1f - h * pad_y)
                    fx2 = min(float(frame_width), x2f + w * pad_x)
                    fy2 = min(float(frame_height), y2f + h * pad_y)
                    ix1, iy1, ix2, iy2 = int(fx1), int(fy1), int(fx2), int(fy2)
                    crop_w, crop_h = ix2 - ix1, iy2 - iy1
                    crop_bbox = (ix1, iy1, crop_w, crop_h)
                    print(f"[INFO] Crop bbox with padding (int): {crop_bbox}")
                    frames_crop_dir = tmp_dir / 'frames_crop'
                    # Robust directory creation (avoid race / partial failure)
                    try:
                        frames_crop_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        print(f"[ERROR] Could not create frames_crop dir: {e}")
                        frames_crop_dir = tmp_dir / 'frames_crop_alt'
                        frames_crop_dir.mkdir(parents=True, exist_ok=True)
                    cap = cv2.VideoCapture(str(local_video))
                    idx = 0
                    success, frame = cap.read()
                    while success:
                        if frame is None:
                            success, frame = cap.read()
                            continue
                        crop = frame[iy1:iy2, ix1:ix2]
                        try:
                            cv2.imwrite(str(frames_crop_dir / f"frame_{idx:06d}.png"), crop)
                        except Exception as e:
                            # If write fails, attempt a fallback raw write directory
                            fallback_dir = tmp_dir / 'frames_crop_fallback'
                            fallback_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(fallback_dir / f"frame_{idx:06d}.png"), crop)
                            frames_crop_dir = fallback_dir
                        idx += 1
                        success, frame = cap.read()
                    cap.release()
                    # Guard: if no frames written, fallback to full-frame
                    if idx == 0 or not frames_crop_dir.exists() or len(list(frames_crop_dir.glob('*.png'))) == 0:
                        print("[WARN] YOLO crop produced no frames -> fallback full-frame OpenPose")
                        crop_bbox = (0, 0, frame_width, frame_height)
                        frames_dir = tmp_dir / 'frames_full'
                        count = _extract_frames(local_video, frames_dir)
                        run_openpose_on_dir(str(frames_dir), str(output_json_dir), str(output_img_dir / 'out'))
                        debug_crop_info.update({'fallback_full_frame': True, 'int_bbox': list(crop_bbox), 'float_bbox': [0.0, 0.0, float(frame_width), float(frame_height)], 'pass1_frame_count': count})
                    else:
                        print(f"[INFO] Cropped {idx} frames to {crop_w}x{crop_h}")
                        # Final existence check before OpenPose run
                        if not frames_crop_dir.exists():
                            raise RuntimeError(f"frames_crop_dir unexpectedly missing before OpenPose: {frames_crop_dir}")
                        run_openpose_on_dir(str(frames_crop_dir), str(output_json_dir), str(output_img_dir / 'out'))
                        debug_crop_info.update({
                            'yolo_bbox': list(map(float, person_bbox)),
                            'float_bbox': [fx1, fy1, fx2 - fx1, fy2 - fy1],
                            'int_bbox': list(crop_bbox),
                            'cropped_frame_count': idx,
                            'crop_dir': str(frames_crop_dir)
                        })
                else:
                    print("[WARN] YOLO failed → fallback full-frame")
                    crop_bbox = (0, 0, frame_width, frame_height)
                    frames_dir = tmp_dir / 'frames_full'
                    count = _extract_frames(local_video, frames_dir)
                    run_openpose_on_dir(str(frames_dir), str(output_json_dir), str(output_img_dir / 'out'))
                    debug_crop_info.update({'fallback_full_frame': True, 'int_bbox': list(crop_bbox), 'float_bbox': [0.0, 0.0, float(frame_width), float(frame_height)], 'pass1_frame_count': count})

            else:  # two_pass_keypoints
                print("[INFO] Using 'two_pass_keypoints' crop strategy")
                # Pass 1: full-frame OpenPose
                frames_full = tmp_dir / 'frames_full'
                full_count = _extract_frames(local_video, frames_full)
                run_openpose_on_dir(str(frames_full), str(output_json_dir), str(output_img_dir / 'pass1'))

                # Parse first pass JSON to build union bbox from keypoints
                json_pass1 = sorted([p for p in output_json_dir.iterdir() if p.suffix == '.json'])
                xs, ys = [], []
                for jp in json_pass1:
                    try:
                        data = json.load(open(jp, 'r', encoding='utf-8'))
                        people = data.get('people', [])
                        if not people:
                            continue
                        kps = np.array(people[0].get('pose_keypoints_2d', [])).reshape(-1, 3)
                        if kps.size == 0:
                            continue
                        # use all joints regardless of confidence (robust bbox); optionally could filter by c>0.05
                        xs.extend(kps[:, 0].tolist())
                        ys.extend(kps[:, 1].tolist())
                    except Exception:
                        continue
                if xs and ys:
                    x1f, y1f, x2f, y2f = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
                    w, h = x2f - x1f, y2f - y1f
                    fx1 = max(0.0, x1f - w * pad_x)
                    fy1 = max(0.0, y1f - h * pad_y)
                    fx2 = min(float(frame_width), x2f + w * pad_x)
                    fy2 = min(float(frame_height), y2f + h * pad_y)
                    ix1, iy1, ix2, iy2 = int(fx1), int(fy1), int(fx2), int(fy2)
                    crop_w, crop_h = ix2 - ix1, iy2 - iy1
                    crop_bbox = (ix1, iy1, crop_w, crop_h)
                    debug_crop_info['pass1_bbox_float'] = [fx1, fy1, fx2 - fx1, fy2 - fy1]
                    debug_crop_info['pass1_bbox_int'] = list(crop_bbox)
                else:
                    print("[WARN] Pass1 produced no keypoints → fallback full-frame second pass")
                    crop_bbox = (0, 0, frame_width, frame_height)
                    debug_crop_info['pass1_bbox_float'] = [0.0, 0.0, float(frame_width), float(frame_height)]
                    debug_crop_info['pass1_bbox_int'] = list(crop_bbox)

                # Prepare second pass output dirs (clear previous JSONs to keep only second pass)
                second_json_dir = tmp_dir / 'json_second'
                second_json_dir.mkdir(parents=True, exist_ok=True)
                # Crop frames for second pass
                frames_crop_dir = tmp_dir / 'frames_crop'
                frames_crop_dir.mkdir(parents=True, exist_ok=True)
                cap2 = cv2.VideoCapture(str(local_video))
                idx2 = 0
                success2, frame2 = cap2.read()
                ix1, iy1 = crop_bbox[0], crop_bbox[1]
                ix2 = ix1 + crop_bbox[2]
                iy2 = iy1 + crop_bbox[3]
                while success2:
                    crop_frame = frame2[iy1:iy2, ix1:ix2]
                    cv2.imwrite(str(frames_crop_dir / f"frame_{idx2:06d}.png"), crop_frame)
                    idx2 += 1
                    success2, frame2 = cap2.read()
                cap2.release()
                if idx2 == 0:
                    print("[ERROR] Two-pass: no frames cropped; falling back to full-frame for second pass")
                    # fallback: use full frames directory
                    frames_crop_dir = frames_full
                    idx2 = full_count
                print(f"[INFO] Two-pass: cropped {idx2} frames to {crop_bbox[2]}x{crop_bbox[3]}")
                run_openpose_on_dir(str(frames_crop_dir), str(second_json_dir), str(output_img_dir / 'pass2'))

                # Replace output_json_dir contents with second pass JSONs for downstream uniformity
                try:
                    for old in output_json_dir.glob('*.json'):
                        old.unlink()
                    for newj in second_json_dir.glob('*.json'):
                        shutil.copy2(str(newj), str(output_json_dir / newj.name))
                except Exception:
                    pass
                debug_crop_info.update({
                    'pass2_frame_count': idx2,
                    'int_bbox': list(crop_bbox),
                    'float_bbox': debug_crop_info.get('pass1_bbox_float'),
                    'cropped_frame_count': idx2
                })

            # persist crop debug info early
            try:
                response_crop_dbg_path = Path(dest_dir) / 'crop_debug.json'
                response_crop_dbg_path.write_text(json.dumps(debug_crop_info, indent=2), encoding='utf-8')
            except Exception:
                pass

            # OpenPose 결과(JSON)를 파싱하여 frames_keypoints 리스트 생성 (로컬 openpose_crop.py와 동일)
            # CRITICAL: JSON still contains COCO-18, must remap to COCO-17
            # CRITICAL: Coordinates are in cropped frame space (matching local training data)
            _IDX_MAP_18_TO_17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
            KP_17 = [
                "Nose", "LEye", "REye", "LEar", "REar",
                "LShoulder", "RShoulder", "LElbow", "RElbow",
                "LWrist", "RWrist", "LHip", "RHip",
                "LKnee", "RKnee", "LAnkle", "RAnkle"
            ]
            COLS_2D = [f"{n}_{a}" for n in KP_17 for a in ("x", "y", "c")]
            
            json_paths = sorted([p for p in output_json_dir.iterdir() if p.suffix == '.json'])
            frames_keypoints = []  # list[frame][joint] -> [x,y,c]
            
            # Parse JSON outputs into frames_keypoints (first person only, matching local)
            for jp in json_paths:
                with jp.open('r', encoding='utf-8') as f:
                    jdata = json.load(f)
                people = jdata.get('people', [])
                if not people:
                    # No person detected → empty frame
                    frames_keypoints.append([])
                else:
                    # Take first person only (matching local training data)
                    kps = people[0].get('pose_keypoints_2d', [])
                    person_18 = np.array(kps).reshape(-1, 3)
                    # Remap COCO-18 to COCO-17
                    if person_18.shape[0] >= 18:
                        person_17 = person_18[_IDX_MAP_18_TO_17, :]
                    else:
                        person_17 = person_18
                    person = [[float(x), float(y), float(c)] for (x, y, c) in person_17]
                    frames_keypoints.append(person)
            
            # CRITICAL: Apply interpolation (matches local preprocessing exactly)
            # NOTE: conf_thresh is now ignored since confidence filtering was disabled
            # All (x,y) coordinates are preserved unless they are (0,0,0)
            interp = interpolate_sequence(
                frames_keypoints,
                conf_thresh=0.0,
                method='linear',
                fill_method='zero',
                limit=None
            )
            
            if not interp:
                raise RuntimeError('Interpolated sequence empty')
            
            # CRITICAL: Transform cropped coordinates back to original frame space
            # Local preprocessing saves coordinates in original frame space, not cropped
            # This fixes the 2px difference between local.csv and skeleton2d.csv
            crop_x_offset = crop_bbox[0] if crop_bbox else 0
            crop_y_offset = crop_bbox[1] if crop_bbox else 0
            
            interp_transformed = []
            for frame in interp:
                transformed_frame = []
                for kp in frame:
                    # Add crop offset to x, y coordinates (keep confidence as-is)
                    transformed_frame.append([
                        kp[0] + crop_x_offset,
                        kp[1] + crop_y_offset,
                        kp[2]
                    ])
                interp_transformed.append(transformed_frame)
            
            # Build DataFrame with COCO-17 column names (matching local CSV format)
            n_joints = len(interp_transformed[0]) if interp_transformed else 17
            cols = COLS_2D if len(COLS_2D) == n_joints * 3 else [f'x_{j}' if k%3==0 else (f'y_{j//3}' if k%3==1 else f'c_{j//3}') for j,k in enumerate(range(n_joints*3))]
            
            rows = []
            for frame in interp_transformed:
                flat = []
                for kp in frame:
                    flat.extend([kp[0], kp[1], kp[2]])
                if len(flat) < n_joints * 3:
                    flat.extend([0.0] * (n_joints * 3 - len(flat)))
                rows.append(flat)
            
            df_2d_wide = pd.DataFrame(rows, columns=cols[:n_joints*3])
            
            # Apply Butterworth smoothing to 2D coordinates
            try:
                from metric_algorithm.smoothing import smooth_skeleton_wide
                # Apply low-pass filter with cutoff 0.1, order=2, fps from video metadata
                smoothing_fps = fps if fps is not None else 30.0
                df_2d_wide_smooth = smooth_skeleton_wide(df_2d_wide, order=2, cutoff=0.1, fps=smoothing_fps, dimension='2d')
                df_2d_wide = df_2d_wide_smooth
                print("[INFO] Applied Butterworth smoothing to 2D skeleton")
            except Exception as e:
                print(f"[WARN] 2D smoothing failed: {e}")
                traceback.print_exc()
            
            # CRITICAL: Save interpolated CSV (matching local training data format)
            csv_dir = Path(dest_dir) / 'csv'
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f'{job_id}_skeleton2d.csv'
            df_2d_wide.to_csv(csv_path, index=False)
            print(f"[INFO] Saved interpolated skeleton CSV: {csv_path}")
            
            # CRITICAL: Build tidy format df_2d for metric_algorithm compatibility
            # metric_algorithm expects (frame, person_idx, joint_idx, x, y, conf) format
            tidy_rows = []
            for frame_idx, frame in enumerate(interp_transformed):
                for joint_idx, (x, y, c) in enumerate(frame):
                    tidy_rows.append({
                        'frame': frame_idx,
                        'person_idx': 0,  # Always first person (matching local)
                        'joint_idx': joint_idx,
                        'x': x,
                        'y': y,
                        'conf': c
                    })
            df_2d = pd.DataFrame(tidy_rows)
            
            # Use transformed interpolated data for result_by_frame (already interpolated, no need to re-interpolate later!)
            result_by_frame = [[frame] for frame in interp_transformed]
            
            # 원본 RGB 프레임을 dest_dir/img로 추출하여 메트릭 오버레이가 실제 프레임에 그려지도록 합니다.
            try:
                dest_img_dir = Path(dest_dir) / 'img'
                dest_img_dir.mkdir(parents=True, exist_ok=True)
                if 'local_video' in locals() and Path(local_video).exists():
                    try:
                        import cv2 as _cv2
                        cap = _cv2.VideoCapture(str(local_video))
                        idx = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            outp = dest_img_dir / f"{idx:06d}.png"
                            try:
                                _cv2.imwrite(str(outp), frame)
                            except Exception:
                                pass
                            idx += 1
                        cap.release()
                    except Exception:
                        # If OpenCV not available or extraction fails, fall back to copying
                        # OpenPose rendered images later (there is a fallback further down).
                        pass
            except Exception:
                pass
        elif dimension == '3d':  # --- 3D 처리 경로 ---
                # download zip and extract expected color/ and depth/ folders
                local_zip = tmp_dir / 'input.zip'
                
                if is_local:
                    print(f"[DEBUG] 로컬 ZIP 파일 복사 중...")
                    local_input_zip = dest_dir / 'local_input.zip'
                    if not local_input_zip.exists():
                        raise RuntimeError(f"로컬 파일 모드인데 local_input.zip을 찾을 수 없습니다: {local_input_zip}")
                    import shutil
                    shutil.copy(str(local_input_zip), str(local_zip))
                    print(f"[DEBUG] 복사 완료: {local_zip}")
                else:
                    print(f"[DEBUG] S3에서 ZIP 다운로드 중: {bucket}/{key}")
                    s3.download_file(bucket, key, str(local_zip))
                    print(f"[DEBUG] 다운로드 완료: {local_zip}")
                    
                    # Also save a copy to dest_dir/local_input.zip for debugging/inspection
                    try:
                        import shutil
                        local_input_zip = dest_dir / 'local_input.zip'
                        shutil.copy2(str(local_zip), str(local_input_zip))
                        print(f"[DEBUG] S3 파일 복사본 저장: {local_input_zip}")
                    except Exception as e:
                        print(f"[WARN] S3 파일 복사본 저장 실패: {e}")

                # 안전한 압축 해제(zip-slip 방지)와 간단한 용량 제한 처리
                MAX_FILES = 5000
                MAX_TOTAL_UNCOMPRESSED = 1_000_000_000  # bytes
                total_uncompressed = 0
                file_count = 0
                with zipfile.ZipFile(local_zip, 'r') as zf:
                    for zi in zf.infolist():
                        file_count += 1
                        if file_count > MAX_FILES:
                            raise RuntimeError('Zip contains too many files')
                        # Protect against zip-slip
                        extracted_path = tmp_dir / zi.filename
                        if not str(extracted_path.resolve()).startswith(str(tmp_dir.resolve())):
                            raise RuntimeError('Zip contains unsafe path')
                        total_uncompressed += zi.file_size
                        if total_uncompressed > MAX_TOTAL_UNCOMPRESSED:
                            raise RuntimeError('Zip total size too large')
                    zf.extractall(path=tmp_dir)

                # color/ 와 depth/ 디렉터리가 있어야 합니다(대소문자 유연 처리 포함).
                color_dir = tmp_dir / 'color'
                depth_dir = tmp_dir / 'depth'
                if not color_dir.exists() or not depth_dir.exists():
                    # attempt to find them case-insensitively
                    dirs = {p.name.lower(): p for p in tmp_dir.iterdir() if p.is_dir()}
                    color_dir = dirs.get('color', color_dir)
                    depth_dir = dirs.get('depth', depth_dir)
                if not color_dir.exists() or not depth_dir.exists():
                    raise RuntimeError('Expected color/ and depth/ folders in zip')

                # Build a sorted list of depth files as a robust fallback mapping
                depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix == '.npy'])
                depth_count = len(depth_files)
                depth_map_info = {'depth_count': depth_count}

                # 트리 전체에서 intrinsics를 탐색하여 카메라 파라미터/스케일 정보를 사용합니다.
                intr = None
                intr_source = None
                try:
                    # look in obvious places first, then do a recursive search
                    intr_candidates = [tmp_dir / 'intrinsics.json', color_dir / 'intrinsics.json', depth_dir / 'intrinsics.json']
                    if not any([c.exists() for c in intr_candidates]):
                        for cand in tmp_dir.rglob('*.json'):
                            try:
                                txt = cand.read_text(encoding='utf-8')
                                if 'cx' in txt and 'fx' in txt:
                                    intr_candidates.append(cand)
                                    break
                            except Exception:
                                continue
                    for cand in intr_candidates:
                        if not cand or not Path(cand).exists():
                            continue
                        try:
                            intr_full_try = json.loads(Path(cand).read_text(encoding='utf-8'))
                            for key in ('color_intrinsics', 'intrinsics', 'intrinsics_color', 'camera_intrinsics'):
                                if key in intr_full_try:
                                    intr = intr_full_try.get(key)
                                    intr_source = str(cand)
                                    intr_full = intr_full_try
                                    break
                            if intr is None and isinstance(intr_full_try, dict) and all(k in intr_full_try for k in ('cx', 'cy', 'fx', 'fy')):
                                intr = intr_full_try
                                intr_source = str(cand)
                                intr_full = intr_full_try
                            if intr is not None:
                                break
                        except Exception:
                            continue
                except Exception:
                    intr = None

                # 깊이 파일 샘플 검증(최대 10개): dtype/shape/유효 픽셀 등 점검
                depth_validation = []
                try:
                    if depth_count == 0:
                        # persist validation info for debugging
                        (Path(dest_dir) / 'depth_validation.json').write_text(json.dumps({'error': 'no_npy_files'}, indent=2))
                        raise RuntimeError('No depth .npy files found in depth directory')

                    # choose up to 10 samples evenly spaced
                    sample_n = min(10, depth_count)
                    if sample_n <= 0:
                        sample_idxs = []
                    else:
                        step = max(1, depth_count // sample_n)
                        sample_idxs = list(range(0, depth_count, step))[:sample_n]

                    invalid_samples = 0
                    for idx in sample_idxs:
                        p = depth_files[idx]
                        stat = {'name': p.name}
                        try:
                            arr = np.load(str(p))
                            stat['dtype'] = str(arr.dtype)
                            stat['shape'] = arr.shape
                            total = int(arr.size)
                            finite = np.isfinite(arr)
                            finite_count = int(finite.sum())
                            valid = finite & (arr > 0)
                            valid_count = int(valid.sum())
                            stat['total_pixels'] = total
                            stat['finite_count'] = finite_count
                            stat['valid_count'] = valid_count
                            stat['min'] = float(np.nanmin(arr[finite])) if finite.any() else None
                            stat['max'] = float(np.nanmax(arr[finite])) if finite.any() else None
                            stat['median_valid'] = float(np.nanmedian(arr[valid])) if valid.any() else None
                            if valid_count == 0:
                                invalid_samples += 1
                        except Exception as e:
                            stat['error'] = str(e)
                            invalid_samples += 1
                        depth_validation.append(stat)

                    # persist validation info for debugging
                    try:
                        (Path(dest_dir) / 'depth_validation.json').write_text(json.dumps(depth_validation, indent=2))
                    except Exception:
                        pass

                    # 깊이 스케일 추론(휴리스틱)
                    # - depth 데이터의 dtype과 범위로 판정 (우선)
                    # - fallback: intrinsics.meta.depth_scale
                    depth_scale_factor = 1.0
                    inferred_unit = 'meters'
                    fps = None  # Extract fps from intrinsics meta for metrics pipeline
                    try:
                        # 1단계: depth dtype과 범위로 판정 (depth 데이터 자체의 단위)
                        if depth_validation and len(depth_validation) > 0:
                            first_depth_info = depth_validation[0]
                            max_val = first_depth_info.get('max', 0)
                            dtype_str = first_depth_info.get('dtype', 'float32')
                            
                            # float32/float64 with small values (< 100) → already meters
                            if 'float' in dtype_str.lower() and max_val < 100:
                                depth_scale_factor = 1.0
                                inferred_unit = 'already_meters'
                            # int16/uint16 with large values (> 1000) → millimeters
                            elif 'int' in dtype_str.lower() and max_val > 1000:
                                depth_scale_factor = 0.001
                                inferred_unit = 'millimeters_to_meters'
                            print(f"[INFO] Depth scale inferred from data: dtype={dtype_str}, max={max_val}, scale={depth_scale_factor}")
                    except Exception as e:
                        print(f"[WARN] Failed to infer depth scale from validation: {e}")
                    
                    try:
                        if 'intr_full' in locals() and isinstance(intr_full, dict):
                            meta = intr_full.get('meta') or intr_full.get('meta_data') or {}
                            if isinstance(meta, dict):
                                # Extract fps if present (used by metrics modules for overlay MP4 generation)
                                if 'fps' in meta:
                                    try:
                                        fps = int(meta.get('fps'))
                                        print(f"[INFO] Extracted fps={fps} from intrinsics.json meta")
                                    except Exception as e:
                                        print(f"[WARN] Failed to parse fps from meta: {e}")
                    except Exception:
                        pass
                    
                    # CRITICAL FIX: If fps is still None after intrinsics extraction, estimate from frame count
                    # This ensures metrics modules always have a valid fps for overlay generation
                    if fps is None and depth_count > 0:
                        # Estimate fps: assume 60fps for depth-based recordings (common for RealSense D455)
                        # This is a reasonable default that matches typical depth recorder fps
                        fps = 60
                        print(f"[INFO] fps not found in intrinsics, using default={fps} (based on depth frame count={depth_count})")
                    elif fps is None:
                        # Final fallback: use 30fps
                        fps = 30
                        print(f"[INFO] fps not found, using final fallback={fps}")

                    # record inferred scale into depth_map_info for diagnostics
                    try:
                        depth_map_info['inferred_depth_scale'] = depth_scale_factor
                        depth_map_info['inferred_unit'] = inferred_unit
                        if intr_source:
                            depth_map_info['intrinsic_source'] = intr_source
                        (Path(dest_dir) / 'depth_map_info.json').write_text(json.dumps(depth_map_info, indent=2))
                    except Exception:
                        pass

                    # If all sampled depth files are invalid, bail out early with an error log
                    if len(sample_idxs) > 0 and invalid_samples >= len(sample_idxs):
                        err_msg = f"All sampled depth .npy files appear invalid (depth_count={depth_count}). Aborting 3D processing."
                        try:
                            (Path(dest_dir) / 'result_error.txt').write_text(err_msg)
                        except Exception:
                            pass
                        return {'message': 'ERROR', 'detail': err_msg}
                except Exception as e:
                    # If validation discovery itself fails, log and continue to attempt processing
                    try:
                        (Path(dest_dir) / 'depth_validation_error.txt').write_text(traceback.format_exc())
                    except Exception:
                        pass

                # color 디렉터리에 대해 OpenPose 실행
                # openpose.openpose는 --write_images 경로로 dirname을 사용하므로,
                # output_img_dir 하위 파일 경로를 인자로 넘겨 해당 폴더에 이미지가 저장되도록 합니다.
                run_openpose_on_dir(str(color_dir), str(output_json_dir), str(output_img_dir / 'out'))

                # Pair JSON outputs with depth .npy by frame index ordering and build DataFrame
                # CRITICAL: JSON still contains COCO-18, must remap to COCO-17
                _IDX_MAP_18_TO_17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
                json_paths = sorted([p for p in output_json_dir.iterdir() if p.suffix == '.json'])
                
                # CRITICAL: Apply early interpolation (matching 2D path)
                # Parse JSON to build frames_keypoints FIRST, then interpolate (STGCN 입력 통일)
                frames_keypoints_3d = []  # list[frame][joint] -> [x,y,c]
                for jp in json_paths:
                    with jp.open('r', encoding='utf-8') as f:
                        jdata = json.load(f)
                    raw_people = jdata.get('people', [])
                    if not raw_people:
                        # No person detected → empty frame
                        frames_keypoints_3d.append([])
                    else:
                        # Take first person only (matching 2D path)
                        kps = raw_people[0].get('pose_keypoints_2d', [])
                        person_18 = np.array(kps).reshape(-1, 3)
                        # Remap COCO-18 to COCO-17
                        if person_18.shape[0] >= 18:
                            person_17 = person_18[_IDX_MAP_18_TO_17, :]
                        else:
                            person_17 = person_18
                        person = [[float(x), float(y), float(c)] for (x, y, c) in person_17]
                        frames_keypoints_3d.append(person)
                
                # Apply interpolation (matching 2D path exactly)
                # NOTE: conf_thresh is now ignored since confidence filtering was disabled
                # All (x,y) coordinates are preserved unless they are (0,0,0)
                interp_3d = interpolate_sequence(
                    frames_keypoints_3d,
                    conf_thresh=0.0,
                    method='linear',
                    fill_method='zero',
                    limit=None
                )
                
                if not interp_3d:
                    raise RuntimeError('Interpolated 3D sequence empty')
                
                # Convert interpolated sequence to result_by_frame format (for consistency)
                result_by_frame = [[frame] for frame in interp_3d]
                
                rows = []
                # Build a sorted list of depth files as a robust fallback mapping
                depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix == '.npy'])
                depth_count = len(depth_files)
                depth_map_info = {'depth_count': depth_count}



                # Try to locate intrinsics in common locations and accept multiple key names
                intr = None
                intr_search_candidates = [tmp_dir / 'intrinsics.json', color_dir / 'intrinsics.json', depth_dir / 'intrinsics.json']
                for cand in intr_search_candidates:
                    if cand.exists():
                        try:
                            intr_full = json.loads(cand.read_text(encoding='utf-8'))
                            # common key names used by recorder tools
                            for key in ('color_intrinsics', 'intrinsics', 'intrinsics_color', 'camera_intrinsics'):
                                if key in intr_full:
                                    intr = intr_full.get(key)
                                    intr_source = str(cand)
                                    break
                            # fallback: if the json itself looks like the intrinsics dict
                            if intr is None and isinstance(intr_full, dict) and all(k in intr_full for k in ('cx', 'cy', 'fx', 'fy')):
                                intr = intr_full
                                intr_source = str(cand)
                            if intr is not None:
                                break
                        except Exception:
                            # ignore parse errors and continue searching
                            intr = None

                # Save a small debug listing into dest_dir so operators can inspect
                try:
                    dbg = {
                        'color_files': [p.name for p in sorted(color_dir.iterdir())[:50]],
                        'depth_files': [p.name for p in depth_files[:50]],
                        'intrinsic_source': intr_source if 'intr_source' in locals() else (intr_source if 'intr_source' not in locals() and 'intr_source' in globals() else None),
                    }
                    # write to dest_dir (persist across tmpdir removal)
                    try:
                        (Path(dest_dir) / 'debug_file_listing.json').write_text(json.dumps(dbg, indent=2))
                    except Exception:
                        pass
                except Exception:
                    pass

                # per-frame debug entries to help diagnose NaN Z values
                frame_depth_debug = []

                for frame_idx, (jp, interpolated_frame) in enumerate(zip(json_paths, interp_3d)):
                    # depth file expected to be zero-padded with same index (e.g., 000001.npy)
                    depth_path = depth_dir / f"{frame_idx:06d}.npy"
                    if not depth_path.exists():
                        # try without zero-pad pattern match
                        depth_candidates = sorted(depth_dir.glob(f"*{frame_idx}*.npy"))
                        if depth_candidates:
                            depth_path = depth_candidates[0]
                        else:
                            # fallback: if number of depth files equals number of json frames or at least > frame_idx,
                            # map by sorted order (assume same ordering)
                            if depth_count > frame_idx:
                                depth_path = depth_files[frame_idx]
                                depth_map_info[f'frame_{frame_idx}'] = f'used_index_map:{depth_path.name}'
                            else:
                                # explicit error with debug info
                                raise FileNotFoundError(f"Depth not found for frame {frame_idx} in {depth_dir} (depth_count={depth_count})")
                    # load depth array; guard against malformed files
                    try:
                        depth_m = np.load(str(depth_path))
                    except Exception:
                        # register debug info and continue (Z will be NaN)
                        depth_m = None

                    # Use interpolated 2D coordinates (already processed and interpolated)
                    # Extract from interpolated_frame which is [x, y, c] for each joint
                    person = interpolated_frame  # This is already the COCO-17 format with interpolation applied
                    ppl = [person]  # Wrap in list to match result_by_frame format

                    # compute 3D coordinates using intrinsics (found earlier in the tmp tree)
                    # NOTE: do not reassign `intr` here; it was discovered earlier and should persist
                    # joint_idx here refers to COCO-17 index
                    for person_idx, person in enumerate(ppl):
                        for joint_idx, (x, y, c) in enumerate(person):
                            X, Y, Zm = (np.nan, np.nan, np.nan)
                            # Initialize 3D coordinates
                            X, Y, Zm = np.nan, np.nan, np.nan
                            
                            sample_info = {
                                'frame': frame_idx,
                                'json_name': f'{frame_idx:06d}.json',
                                'depth_name': depth_path.name if depth_path is not None else None,
                                'person_idx': person_idx,
                                'joint_idx': joint_idx,
                                'x': float(x),
                                'y': float(y),
                                'conf': float(c),
                                'intrinsics_present': bool(intr),
                                'patch_shape': None,
                                'vals_count': 0,
                                'Z_median': None,
                            }
                            if intr is not None and depth_m is not None:
                                xi, yi = int(round(x)), int(round(y))
                                # robust depth sampling (patch median)
                                try:
                                    H, W = depth_m.shape[:2]
                                    r = 2
                                    x0, x1 = max(0, xi-r), min(W, xi+r+1)
                                    y0, y1 = max(0, yi-r), min(H, yi+r+1)
                                    patch = depth_m[y0:y1, x0:x1]
                                    sample_info['patch_shape'] = patch.shape
                                    vals = patch[np.isfinite(patch) & (patch > 0)]
                                    sample_info['vals_count'] = int(vals.size)
                                    Z_raw = float(np.median(vals)) if vals.size else np.nan
                                    # depth_scale_factor를 사용해서 미터 단위로 변환
                                    if np.isnan(Z_raw):
                                        Z_meter = np.nan
                                    elif depth_scale_factor == 1.0:
                                        Z_meter = Z_raw  # 이미 미터 단위
                                    else:
                                        Z_meter = Z_raw * depth_scale_factor
                                    
                                    # CSV에 저장할 때는 미터 단위 (swing_speed.py에서 scale_to_m=1.0으로 처리)
                                    sample_info['Z_median'] = None if np.isnan(Z_meter) else float(Z_meter)
                                    
                                    # [DEBUG] 처음 5프레임만 로그 출력
                                    if frame_idx < 5:
                                        print(f"[DEBUG] Frame {frame_idx}, Joint {joint_idx}: Z_raw={Z_raw:.4f}, depth_scale={depth_scale_factor}, Z_meter={Z_meter:.4f}, sample['Z_median']={sample_info['Z_median']}")
                                    
                                    # Z가 유효하면 X, Y도 계산 (미터 단위 Z 사용)
                                    if sample_info['Z_median'] is not None:
                                        Z_meter_val = float(sample_info['Z_median'])  # 미터 단위
                                        try:
                                            cx = float(intr.get('cx', 320))
                                            cy = float(intr.get('cy', 180))
                                            fx = float(intr.get('fx', 320))
                                            fy = float(intr.get('fy', 180))
                                            # X/Y를 미터 단위로 계산 (unprojection: 표준 공식)
                                            X = (x - cx) * Z_meter_val / fx
                                            Y = (y - cy) * Z_meter_val / fy
                                            if frame_idx < 5:
                                                print(f"[DEBUG] Frame {frame_idx}, Joint {joint_idx}: X={X:.4f}m, Y={Y:.4f}m, Z={Z_meter_val:.4f}m")
                                        except Exception as e:
                                            if frame_idx < 5:
                                                print(f"[DEBUG] Frame {frame_idx}, Joint {joint_idx}: X/Y 계산 실패: {e}")
                                            pass
                                    elif frame_idx < 5:
                                        print(f"[DEBUG] Frame {frame_idx}, Joint {joint_idx}: Z_median is None - X/Y 계산 스킵")
                                except Exception as e:
                                    if frame_idx < 5:
                                        print(f"[DEBUG] Frame {frame_idx}, Joint {joint_idx}: Depth 샘플링 실패: {e}")
                                    pass
                            elif frame_idx < 5:
                                print(f"[DEBUG] Frame {frame_idx}, Joint {joint_idx}: intr={intr is not None}, depth_m={depth_m is not None} - 조건 불만족")


                            # append tidy row
                            # [DEBUG] 모든 데이터에 대해 결과 기록
                            x_val = float(X) if not np.isnan(X) else np.nan
                            y_val = float(Y) if not np.isnan(Y) else np.nan
                            z_val = float(Zm) if (not np.isnan(Zm) and Zm is not None and np.isfinite(Zm)) else np.nan
                            
                            rows.append({
                                'frame': frame_idx,
                                'person_idx': person_idx,
                                'joint_idx': joint_idx,
                                'x': float(x), 'y': float(y), 'conf': float(c),
                                'X': x_val,
                                'Y': y_val,
                                'Z': z_val,  # 미터 단위
                            })
                            # record sample debug for first N frames or all (keeps small)
                            if frame_idx < 200:  # limit debug size
                                frame_depth_debug.append(sample_info)

                # persist depth mapping info and per-frame depth debug to dest_dir for diagnostics
                try:
                    dm_path = Path(dest_dir) / 'depth_map_info.json'
                    dm = depth_map_info.copy()
                    dm['sample'] = depth_files[:50]
                    dm_path.write_text(json.dumps(dm, indent=2))
                except Exception:
                    pass

                try:
                    dbg_path = Path(dest_dir) / 'frame_depth_debug.json'
                    dbg_path.write_text(json.dumps(frame_depth_debug, indent=2))
                except Exception:
                    pass

                df_3d = pd.DataFrame(rows)
                
                # 디버그: X, Y, Z가 제대로 저장되었는지 확인
                x_nans = df_3d['X'].isna().sum() if 'X' in df_3d.columns else 'NO COL'
                y_nans = df_3d['Y'].isna().sum() if 'Y' in df_3d.columns else 'NO COL'
                z_nans = df_3d['Z'].isna().sum() if 'Z' in df_3d.columns else 'NO COL'
                print(f"[INFO] df_3d NaN counts (BEFORE filtering): X={x_nans}, Y={y_nans}, Z={z_nans} (total rows={len(df_3d)})")
                
                # CRITICAL: Apply interpolation to df_3d BEFORE filtering+metrics
                # df_3d is in tidy format: frame, person_idx, joint_idx, x, y, conf, X, Y, Z
                # Interpolate each joint's X/Y/Z independently across frames
                if not df_3d.empty and 'Z' in df_3d.columns:
                    try:
                        print("[INFO] Applying interpolation to df_3d (tidy format)...")
                        n_nans_before = df_3d[['X', 'Y', 'Z']].isna().sum().sum()
                        print(f"[DEBUG] df_3d NaN BEFORE interp: X={df_3d['X'].isna().sum()}, Y={df_3d['Y'].isna().sum()}, Z={df_3d['Z'].isna().sum()}")
                        
                        # Group by joint_idx and person_idx, then interpolate each group independently
                        df_3d_interp = df_3d.copy()
                        
                        # Interpolate each joint's coordinates
                        for person_idx in df_3d['person_idx'].unique():
                            for joint_idx in df_3d['joint_idx'].unique():
                                mask = (df_3d_interp['person_idx'] == person_idx) & (df_3d_interp['joint_idx'] == joint_idx)
                                if not mask.any():
                                    continue
                                
                                # Get the subset for this joint
                                joint_data = df_3d_interp.loc[mask].copy()
                                
                                # Interpolate X, Y, Z
                                for col in ['X', 'Y', 'Z']:
                                    if col in joint_data.columns:
                                        # Linear interpolate → forward fill → backward fill → zero fill
                                        joint_data[col] = joint_data[col].interpolate(method='linear', limit=None, limit_direction='both')
                                        joint_data[col] = joint_data[col].ffill(limit=None)
                                        joint_data[col] = joint_data[col].bfill(limit=None)
                                        joint_data[col] = joint_data[col].fillna(0.0)
                                
                                # Update df_3d_interp with interpolated values
                                df_3d_interp.loc[mask, ['X', 'Y', 'Z']] = joint_data[['X', 'Y', 'Z']]
                        
                        df_3d = df_3d_interp
                        n_nans_after = df_3d[['X', 'Y', 'Z']].isna().sum().sum()
                        print(f"[INFO] df_3d interpolation complete: NaN count {n_nans_before} → {n_nans_after}")
                        print(f"[DEBUG] df_3d NaN AFTER interp: X={df_3d['X'].isna().sum()}, Y={df_3d['Y'].isna().sum()}, Z={df_3d['Z'].isna().sum()}")
                    except Exception as e:
                        print(f"[WARN] df_3d interpolation failed: {e}")
                        traceback.print_exc()
                
                # 3D 필터링 DISABLED - 필터링이 새로운 NaN을 생성하여 보간 효과를 무효화
                # 깊이값 샘플링(median) 단계에서 이미 outlier가 제거되었으므로 필터링 불필요
                # NOTE: 나중에 metrics 계산 중에 추가 필터링이 필요하면 metric 모듈 내에서 수행
                #
                # if not df_3d.empty and 'Z' in df_3d.columns:
                #     try:
                #         df_3d = filter_z_outliers_by_frame_delta(df_3d, relative_threshold=0.5)
                #         ...
                #     except Exception as e:
                #         print(f"[WARN] Frame delta filtering failed: {e}")
                #     
                #     try:
                #         df_3d = filter_z_outliers_by_frame_consistency(df_3d, ...)
                #         ...
                #     except Exception as e:
                #         print(f"[WARN] Frame consistency filtering failed: {e}")

                # Build a 2D tidy DataFrame (frame, person_idx, joint_idx, x, y, conf)
                # using the parsed OpenPose JSON (result_by_frame) so overlay uses pure 2D pixel coords.
                try:
                    rows2 = []
                    for frame_idx, ppl in enumerate(result_by_frame):
                        for person_idx, person in enumerate(ppl):
                            for joint_idx, (x, y, c) in enumerate(person):
                                rows2.append({'frame': frame_idx, 'person_idx': person_idx, 'joint_idx': joint_idx, 'x': float(x), 'y': float(y), 'conf': float(c)})
                    df_2d = pd.DataFrame(rows2)
                except Exception:
                    df_2d = pd.DataFrame()

                # Also produce 'wide' DataFrames expected by metric modules: wide2 (2D pixels) and wide3 (3D X/Y/Z)
                try:
                    from metric_algorithm.runner_utils import tidy_to_wide
                    wide2 = tidy_to_wide(df_2d, dimension='2d', person_idx=0) if (not df_2d.empty) else pd.DataFrame()
                except Exception:
                    wide2 = pd.DataFrame()

                # CRITICAL: Convert df_3d (already interpolated) to wide format ONCE
                # wide3 is used both for CSV output AND for metrics - atomic operation
                try:
                    from metric_algorithm.runner_utils import tidy_to_wide
                    # df_3d is already interpolated and filtered above, just convert to wide format
                    wide3 = tidy_to_wide(df_3d, dimension='3d', person_idx=0) if (not df_3d.empty) else pd.DataFrame()
                    
                    # [DEBUG] 보간 후 wide3 NaN 상태 확인
                    print(f"[DEBUG] After tidy_to_wide conversion:")
                    print(f"[DEBUG]   wide3 shape: {wide3.shape}")
                    print(f"[DEBUG]   wide3 total NaN count: {wide3.isna().sum().sum()}")
                    if not wide3.empty:
                        # Check specific joint columns
                        shoulder_cols = [c for c in wide3.columns if 'Shoulder' in c]
                        hip_cols = [c for c in wide3.columns if 'Hip' in c]
                        print(f"[DEBUG]   LShoulder NaN: {wide3[[c for c in shoulder_cols if 'LShoulder' in c]].isna().sum().sum() if shoulder_cols else 'N/A'}")
                        print(f"[DEBUG]   RShoulder NaN: {wide3[[c for c in shoulder_cols if 'RShoulder' in c]].isna().sum().sum() if shoulder_cols else 'N/A'}")
                        print(f"[DEBUG]   LHip NaN: {wide3[[c for c in hip_cols if 'LHip' in c]].isna().sum().sum() if hip_cols else 'N/A'}")
                        print(f"[DEBUG]   RHip NaN: {wide3[[c for c in hip_cols if 'RHip' in c]].isna().sum().sum() if hip_cols else 'N/A'}")
                        
                        # Check first few rows
                        try:
                            print(f"[DEBUG] wide3 first row: {wide3.iloc[0].to_dict() if len(wide3) > 0 else 'empty'}")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[ERROR] tidy_to_wide conversion failed: {e}")
                    traceback.print_exc()
                    wide3 = pd.DataFrame()

                # CRITICAL: Save 3D skeleton CSV (matching 2D skeleton2d.csv format)
                # Save skeleton3d.csv with wide3 format (frame-by-frame, COCO joints with X/Y/Z coordinates)
                # NOTE: wide3 comes from already-interpolated df_3d, so no duplicate interpolation
                try:
                    csv_dir = Path(dest_dir) / 'csv'
                    csv_dir.mkdir(parents=True, exist_ok=True)
                    csv_3d_path = csv_dir / f'{job_id}_skeleton3d.csv'
                    if isinstance(wide3, pd.DataFrame) and not wide3.empty:
                        wide3.to_csv(csv_3d_path, index=False)
                        print(f"[INFO] Saved 3D skeleton CSV: {csv_3d_path}")
                    else:
                        print(f"[WARN] wide3 DataFrame is empty or not valid, skipping skeleton3d.csv write")
                except Exception as e:
                    print(f"[ERROR] Failed to save skeleton3d.csv: {e}")
                    traceback.print_exc()

                # Persist original RGB frames into dest_dir/img BEFORE tmpdir is removed so operators
                # and metric modules can access them. For 2D input we extract frames from the input MP4;
                # for 3D input we copy the recorder's color/ images. Keep OpenPose rendered images (if any)
                # available under dest_dir/openpose_img for debugging.
                try:
                    dest_img_dir = Path(dest_dir) / 'img'
                    dest_img_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[DEBUG] Created dest_img_dir: {dest_img_dir}")

                    # Prefer original color frames (if present). For 3D runs color_dir contains originals.
                    try:
                        # If color_dir exists in tmp tree, copy those images as the canonical RGB frames
                        if 'color_dir' in locals() and Path(color_dir).exists():
                            print(f"[DEBUG] Attempting to copy from color_dir: {color_dir}")
                            color_files = sorted([p for p in Path(color_dir).iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
                            print(f"[DEBUG] Found {len(color_files)} color image files in {color_dir}")
                            copied_count = 0
                            for p in color_files:
                                try:
                                    shutil.copy2(str(p), str(dest_img_dir / p.name))
                                    copied_count += 1
                                except Exception as e:
                                    print(f"[WARN] Failed to copy {p.name}: {e}")
                            print(f"[DEBUG] Successfully copied {copied_count}/{len(color_files)} color images to {dest_img_dir}")
                        else:
                            print(f"[DEBUG] color_dir not available or doesn't exist (color_dir in locals: {'color_dir' in locals()}, exists: {Path(color_dir).exists() if 'color_dir' in locals() else 'N/A'})")
                    except Exception as e:
                        print(f"[ERROR] Exception while copying color_dir images: {e}")
                        traceback.print_exc()

                    # If we have an input mp4 (2D path), try to extract frames into dest/img so overlays draw on RGB
                    try:
                        if 'local_video' in locals() and Path(local_video).exists():
                            try:
                                import cv2 as _cv2
                                cap = _cv2.VideoCapture(str(local_video))
                                idx = 0
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    outp = dest_img_dir / f"{idx:06d}.png"
                                    try:
                                        _cv2.imwrite(str(outp), frame)
                                    except Exception:
                                        # fallback: skip frame write
                                        pass
                                    idx += 1
                                cap.release()
                            except Exception:
                                # if OpenCV missing or fails, fall back to copying OpenPose images below
                                pass
                    except Exception:
                        pass

                    # If no original frames were copied/extracted, fall back to copying OpenPose images
                    # but DO NOT copy OpenPose-rendered images (they are stored separately in openpose_img).
                    try:
                        has_rgb = any(p for p in dest_img_dir.iterdir())
                    except Exception:
                        has_rgb = False
                    if not has_rgb and output_img_dir.exists():
                        for p in sorted(output_img_dir.iterdir()):
                            try:
                                if p.is_file():
                                    lname = p.name.lower()
                                    # skip rendered/debug images (these are persisted under openpose_img)
                                    if 'render' in lname or 'openpose' in lname:
                                        continue
                                    shutil.copy2(str(p), str(dest_img_dir / p.name))
                            except Exception:
                                pass

                    # Also persist OpenPose rendered images into dest_dir/openpose_img for debugging
                    try:
                        openpose_dest = Path(dest_dir) / 'openpose_img'
                        openpose_dest.mkdir(parents=True, exist_ok=True)
                        if output_img_dir.exists():
                            for p in sorted(output_img_dir.iterdir()):
                                try:
                                    if p.is_file():
                                        shutil.copy2(str(p), str(openpose_dest / p.name))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    pass

                # If no overlay mp4 yet, create a simple mp4 from copied images (written into dest_dir)
                try:
                    # Prefer using original color frames if available (avoid using OpenPose-rendered images)
                    imgs = []
                    src_desc = 'none'
                    try:
                        print(f"[DEBUG] Starting overlay MP4 generation for dimension={dimension}")
                        print(f"[DEBUG]   color_dir variable check: {'color_dir' in locals()}, exists: {Path(color_dir).exists() if 'color_dir' in locals() else 'N/A'}")
                        print(f"[DEBUG]   dest_img_dir: {dest_img_dir}, exists: {dest_img_dir.exists()}")
                        
                        # prefer color_dir (original frames) when present
                        if 'color_dir' in locals() and Path(color_dir).exists():
                            imgs = sorted([p for p in Path(color_dir).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                            src_desc = 'color_dir'
                            print(f"[DEBUG]   Using color_dir: found {len(imgs)} images")
                        else:
                            if dest_img_dir.exists():
                                imgs = sorted([p for p in dest_img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                                src_desc = 'dest_img_dir'
                                print(f"[DEBUG]   Using dest_img_dir: found {len(imgs)} images")
                            else:
                                print(f"[DEBUG]   dest_img_dir does not exist: {dest_img_dir}")
                        
                        # deduplicate by name while preserving order
                        seen = set()
                        uniq = []
                        for p in imgs:
                            if p.name in seen:
                                continue
                            seen.add(p.name)
                            uniq.append(p)
                        imgs = uniq
                        print(f"[DEBUG]   After dedup: {len(imgs)} unique images from {src_desc}")
                    except Exception as e:
                        print(f"[ERROR] Image collection failed: {e}")
                        import traceback
                        traceback.print_exc()
                        imgs = []
                        src_desc = 'none'

                    if imgs:
                        try:
                            from metric_algorithm.runner_utils import images_to_mp4
                            out_mp4 = Path(dest_dir) / f"{job_id}_overlay.mp4"
                            overlay_fps = fps if fps is not None else 30.0
                            print(f"[INFO] Generating overlay MP4 with {len(imgs)} images, fps={overlay_fps}, output={out_mp4}")
                            created, used = images_to_mp4(imgs, out_mp4, fps=overlay_fps, resize=None, filter_rendered=True, write_debug=True)
                            print(f"[INFO] Overlay MP4 generation result: created={created}, used={used} images")
                            if created:
                                try:
                                    (Path(dest_dir) / 'overlay_debug.json').write_text(json.dumps({'overlay_source': src_desc, 'images_used': used}), encoding='utf-8')
                                except Exception:
                                    pass
                            else:
                                print(f"[WARN] images_to_mp4() returned created=False")
                        except Exception as e:
                            print(f"[ERROR] images_to_mp4 call failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"[WARN] No images found for overlay MP4 generation (src_desc={src_desc})")
                except Exception as e:
                    print(f"[ERROR] Overlay MP4 generation block failed: {e}")
                    import traceback
                    traceback.print_exc()

            # end of tmp work; we intentionally DO NOT remove tmp_dir here so files remain for debugging
            # If automatic cleanup is desired, uncomment the following line to remove the tmp dir:
            # shutil.rmtree(str(tmp_dir))

        # after temp work is done, validate dimension and build interpolated sequence
        if dimension not in ('2d', '3d'):
            raise RuntimeError(f'Unsupported dimension: {dimension}')

        # For 2D, interpolation already applied in the 2D processing block above
        # For 3D, apply interpolation here (3D path doesn't have inline interpolation yet)
        if dimension == '2d':
            # result_by_frame already contains interpolated data from 2D path - use as-is
            sequence = [(ppl[0] if (ppl and len(ppl) > 0) else []) for ppl in result_by_frame]
            people_sequence = [([person] if person else []) for person in sequence]
        else:
            # 3D path: apply interpolation
            sequence = [(ppl[0] if (ppl and len(ppl) > 0) else []) for ppl in result_by_frame]
            # NOTE: conf_thresh is now ignored since confidence filtering was disabled
            interpolated = interpolate_sequence(sequence, conf_thresh=0.0, method='linear', fill_method='zero')
            people_sequence = [([person] if person else []) for person in interpolated]

        response_payload = {
            'message': 'OK',
            'people_sequence': people_sequence,
            'frame_count': len(people_sequence)
        }

        # include dimension and prepare a prefixed basename for all job-derived files
        try:
            response_payload['dimension'] = dimension
        except Exception:
            pass
        
        # CRITICAL: Store crop_bbox info for stgcn_tester to use when converting CSV to PKL
        try:
            if dimension == '2d' and 'crop_bbox' in locals() and crop_bbox is not None:
                response_payload['crop_bbox'] = list(crop_bbox)  # (x, y, w, h)
                print(f"[INFO] Crop bbox stored in response: {crop_bbox}")
        except Exception:
            pass
        
        result_basename = f"{dimension}_{job_id}"

        # Attach minimal DataFrame summary (not full CSV) for downstream verification.
        # DataFrames (df_2d or df_3d) are kept in memory for metrics; here we include simple metadata.
        try:
            if dimension == '2d':
                response_payload['skeleton_rows'] = len(df_2d)
                response_payload['skeleton_columns'] = list(df_2d.columns)
            elif dimension == '3d':
                response_payload['skeleton_rows'] = len(df_3d)
                response_payload['skeleton_columns'] = list(df_3d.columns)
        except Exception:
            pass

        # write a debug/partial JSON early so operators can inspect intermediate state
        partial_out_path = Path(dest_dir) / f"{result_basename}.partial.json"
        out_path = Path(dest_dir) / f"{result_basename}.json"

        # If an async MMACTION thread was started, wait briefly for its response file
        try:
            mmaction_resp_path = None
            dbg = response_payload.get('debug', {})
            # candidates for response path: earlier helpers set mmaction_start.response_path or use standard name
            if isinstance(dbg.get('mmaction_start'), dict) and dbg['mmaction_start'].get('response_path'):
                mmaction_resp_path = Path(dbg['mmaction_start'].get('response_path'))
            # fallback to canonical path in dest_dir (prefixed)
            if mmaction_resp_path is None:
                mmaction_resp_path = Path(dest_dir) / f"{result_basename}_stgcn_response.json"

            # only wait if thread started flag is present
            thread_started = dbg.get('mmaction_thread_started') or dbg.get('mmaction_thread_started_later') or response_payload.setdefault('debug', {}).get('mmaction_thread_started')
            if thread_started:
                # allow override via env var
                try:
                    wait_secs = int(os.environ.get('MMACTION_WAIT_SECONDS', '30'))
                except Exception:
                    wait_secs = 30
                # poll for file existence with small sleep intervals
                import time
                elapsed = 0
                interval = 0.5
                while elapsed < wait_secs:
                    if mmaction_resp_path.exists():
                        break
                    time.sleep(interval)
                    elapsed += interval
                # if response file appeared, merge into response_payload
                try:
                    if mmaction_resp_path.exists():
                        try:
                            _merge_stgcn_resp(mmaction_resp_path, response_payload)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

        try:
            _safe_write_json(partial_out_path, response_payload)
        except Exception:
            # fallback to normal write
            try:
                _safe_write_json(out_path, response_payload)
            except Exception:
                pass

        # Persist OpenPose rendered images into dest_dir/openpose_img for debugging and
        # avoid copying them into dest_dir/img to prevent mixing rendered + original frames.
        try:
            openpose_dest = Path(dest_dir) / 'openpose_img'
            openpose_dest.mkdir(parents=True, exist_ok=True)
            for p in sorted(output_img_dir.iterdir() if output_img_dir.exists() else []):
                try:
                    if p.is_file():
                        # Always copy rendered/openpose images into openpose_img for diagnostics
                        shutil.copy2(str(p), str(openpose_dest / p.name))
                except Exception:
                    pass
        except Exception:
            pass

        # Optionally run local metrics using the in-memory DataFrames; run_metrics_in_process will write CSVs into dest_dir
        metrics_res = None
        try:
            # Debug: Print fps value before passing to metrics
            print(f"[DEBUG] fps before run_metrics_in_process: {fps}")
            print(f"[DEBUG] locals has 'fps' key: {'fps' in locals()}")
            if 'fps' in locals():
                print(f"[DEBUG] locals['fps'] = {locals()['fps']}")
            
            # Prepare context for metrics modules
            # IMPORTANT: Include intrinsics (depth_scale, etc) for swing_speed scale conversion
            ctx_for_metrics = locals().copy()
            
            # CRITICAL: Add img_dir explicitly for metric modules (swing_speed needs this to find overlay images)
            # This is crucial for both 2D and 3D paths to generate overlay MP4s
            if 'dest_img_dir' in locals():
                ctx_for_metrics['img_dir'] = str(dest_img_dir)
                print(f"[DEBUG] ✓ Added img_dir to ctx_for_metrics: {dest_img_dir}")
            
            print(f"[DEBUG] 'intr_full' in locals: {'intr_full' in locals()}")
            if 'intr_full' in locals():
                print(f"[DEBUG] intr_full type: {type(intr_full)}")
                if isinstance(intr_full, dict):
                    print(f"[DEBUG] intr_full keys: {list(intr_full.keys())}")
                    print(f"[DEBUG] intr_full['meta']: {intr_full.get('meta', 'NOT FOUND')}")
            
            if 'intr_full' in locals() and isinstance(intr_full, dict):
                ctx_for_metrics['intrinsics'] = intr_full
                print(f"[DEBUG] ✓ Added intrinsics to ctx_for_metrics: {list(intr_full.keys())}")
                print(f"[DEBUG] ✓ intrinsics.meta.depth_scale = {intr_full.get('meta', {}).get('depth_scale', 'NOT FOUND')}")
            else:
                print(f"[DEBUG] ✗ intrinsics NOT added (intr_full missing or not dict)")
            
            metrics_res = run_metrics_in_process(dimension, ctx_for_metrics)
        except Exception:
            # metrics failures shouldn't take down processing
            traceback.print_exc()

        # If the metric runner produced structured output, merge it into the job result JSON
        try:
            if isinstance(metrics_res, dict):
                # CRITICAL FIX: Extract top-level overlay_mp4 if present (from metric module result)
                top_level_overlay_mp4 = metrics_res.get('overlay_mp4', None)
                
                # Get per-module payload
                metrics_payload = metrics_res.get('metrics') if 'metrics' in metrics_res else metrics_res

                cleaned_metrics = {}
                for mname, mval in (metrics_payload.items() if isinstance(metrics_payload, dict) else []):
                    # If module returned non-dict (e.g., error string), keep as-is
                    if not isinstance(mval, dict):
                        cleaned_metrics[mname] = mval
                        continue

                    cleaned = {}
                    
                    # CRITICAL FIX: If top-level overlay_mp4 exists and module didn't specify its own,
                    # use the top-level one (matches metric module's result structure)
                    if top_level_overlay_mp4 and 'overlay_mp4' not in cleaned:
                        cleaned['overlay_mp4'] = top_level_overlay_mp4

                    # Keep summary if available
                    if 'summary' in mval and isinstance(mval['summary'], dict):
                        cleaned['summary'] = mval['summary']

                    # Collect CSV paths and overlay info from values (strings or lists)
                    csv_paths = []
                    overlay_mp4 = None
                    overlay_s3 = None
                    
                    for k, v in mval.items():
                        # CRITICAL FIX: Collect overlay_mp4/overlay_s3 BEFORE skipping them
                        if k.lower() == 'overlay_mp4' and v:
                            overlay_mp4 = v
                        elif k.lower() == 'overlay_s3' and v:
                            overlay_s3 = v
                        # skip other overlay/mp4 fields during CSV collection
                        elif k.lower().startswith('overlay'):
                            continue
                        elif isinstance(v, str) and v.lower().endswith('.csv'):
                            p = Path(v)
                            if not p.exists():
                                # try relative to dest_dir
                                p = Path(dest_dir) / v
                            if p.exists():
                                csv_paths.append(p)
                        elif isinstance(v, (list, tuple)):
                            for it in v:
                                if isinstance(it, str) and it.lower().endswith('.csv'):
                                    p = Path(it)
                                    if not p.exists():
                                        p = Path(dest_dir) / it
                                    if p.exists():
                                        csv_paths.append(p)

                    # If we found CSVs, read and merge them into frame_data
                    if csv_paths:
                        try:
                            dfs = []
                            for p in csv_paths:
                                try:
                                    df = pd.read_csv(p)
                                except Exception:
                                    # skip unreadable csv
                                    continue
                                # normalize frame column name
                                if 'frame' not in df.columns and 'Frame' in df.columns:
                                    df = df.rename(columns={'Frame': 'frame'})
                                if 'frame' in df.columns:
                                    df = df.set_index('frame')
                                else:
                                    # use row order as frame index
                                    df.index.name = 'frame'
                                dfs.append(df)
                            if dfs:
                                # merge on index (frame)
                                from functools import reduce
                                def _join(a, b):
                                    return a.join(b, how='outer', lsuffix='_l', rsuffix='_r')
                                merged = reduce(_join, dfs) if len(dfs) > 1 else dfs[0]

                                frame_data = {}
                                for idx, row in merged.iterrows():
                                    # cast index to int if possible
                                    try:
                                        fkey = int(idx)
                                    except Exception:
                                        fkey = str(idx)
                                    rowd = {}
                                    for col, val in row.items():
                                        if pd.isna(val):
                                            rowd[col] = None
                                        else:
                                            # convert numpy scalar to python type
                                            try:
                                                if hasattr(val, 'item'):
                                                    val = val.item()
                                            except Exception:
                                                pass
                                            rowd[col] = val
                                    frame_data[str(fkey)] = rowd
                                cleaned['frame_data'] = frame_data
                        except Exception:
                            # non-fatal — include raw csv paths if conversion failed
                            cleaned['frame_data_error'] = 'failed to read/convert csv'

                    # CRITICAL FIX: Always include overlay_mp4 and overlay_s3 if collected
                    if overlay_mp4:
                        cleaned['overlay_mp4'] = overlay_mp4
                    if overlay_s3:
                        cleaned['overlay_s3'] = overlay_s3

                    # If no summary and no frame_data, preserve original (may include error info)
                    if not cleaned:
                        cleaned_metrics[mname] = mval
                    else:
                        cleaned_metrics[mname] = cleaned

                response_payload['metrics'] = cleaned_metrics

                # CRITICAL FIX: Extract top-level overlay_mp4 from first metric that has it
                # This ensures overlay_mp4 appears at the top level of the final JSON
                if top_level_overlay_mp4:
                    response_payload['overlay_mp4'] = top_level_overlay_mp4
                else:
                    # Try to get overlay_mp4 from any metric
                    for mname, ment in cleaned_metrics.items():
                        if isinstance(ment, dict) and ment.get('overlay_mp4'):
                            response_payload['overlay_mp4'] = ment['overlay_mp4']
                            break

                # Ensure every metric entry includes an `overlay_mp4` field when
                # a matching overlay file exists under `dest_dir/mp4/` or `dest_dir/`.
                # Try multiple matching strategies (filename contains metric name,
                # parts of metric name, then fallback to assigning any unclaimed
                # overlay files deterministically).
                try:
                    mp4_dir = Path(dest_dir) / 'mp4'
                    # collect candidate overlay files (mp4s that mention 'overlay' or end with .mp4)
                    candidates = []
                    if mp4_dir.exists():
                        candidates.extend([p for p in sorted(mp4_dir.iterdir()) if p.suffix.lower() == '.mp4'])
                    # also consider mp4s at dest root
                    candidates.extend([p for p in sorted(Path(dest_dir).glob('*.mp4')) if p not in candidates])

                    # only keep overlay-like names first, but keep all mp4s as fallback
                    overlay_like = [p for p in candidates if 'overlay' in p.name.lower()]
                    other_mp4s = [p for p in candidates if p not in overlay_like]
                    candidates = overlay_like + other_mp4s

                    # track which candidate files have been claimed/assigned
                    claimed = set()

                    # helper: normalize strings for comparison
                    def _norm(s: str) -> str:
                        return s.replace('-', '_').lower()

                    # first pass: exact metric name substring match (case-insensitive)
                    for mname, ment in cleaned_metrics.items():
                        try:
                            if not isinstance(ment, dict):
                                continue
                            if ment.get('overlay_mp4'):
                                continue
                            mnorm = _norm(mname)
                            found = None
                            for cand in candidates:
                                if cand in claimed:
                                    continue
                                if mnorm in _norm(cand.name):
                                    found = cand
                                    break
                            if found:
                                ment['overlay_mp4'] = str(Path('mp4') / found.name) if found.parent.samefile(mp4_dir) else str(Path('mp4') / found.name) if mp4_dir.exists() else str(found.name)
                                claimed.add(found)
                        except Exception:
                            pass

                    # second pass: match any token from metric name (split on '_')
                    for mname, ment in cleaned_metrics.items():
                        try:
                            if not isinstance(ment, dict):
                                continue
                            if ment.get('overlay_mp4'):
                                continue
                            tokens = [t for t in _norm(mname).split('_') if t]
                            found = None
                            for cand in candidates:
                                if cand in claimed:
                                    continue
                                cname = _norm(cand.name)
                                if any(tok in cname for tok in tokens):
                                    found = cand
                                    break
                            if found:
                                ment['overlay_mp4'] = str(Path('mp4') / found.name) if found.parent.samefile(mp4_dir) else str(Path('mp4') / found.name)
                                claimed.add(found)
                        except Exception:
                            pass

                    # final fallback: assign any remaining unclaimed overlay-like mp4s
                    for mname, ment in cleaned_metrics.items():
                        try:
                            if not isinstance(ment, dict):
                                continue
                            if ment.get('overlay_mp4'):
                                continue
                            # pick the first unclaimed candidate
                            found = None
                            for cand in candidates:
                                if cand in claimed:
                                    continue
                                found = cand
                                break
                            if found:
                                ment['overlay_mp4'] = str(Path('mp4') / found.name)
                                claimed.add(found)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Final guaranteed fallback: if any metric still lacks overlay_mp4,
                # inject the conventional expected path `mp4/{job_id}_{metric}_overlay.mp4`.
                try:
                    _job = job_id if 'job_id' in locals() else (job if 'job' in locals() else None)
                    if _job:
                        for mname, ment in cleaned_metrics.items():
                            try:
                                if not isinstance(ment, dict):
                                    continue
                                if ment.get('overlay_mp4'):
                                    continue
                                ment['overlay_mp4'] = str(Path('mp4') / f"{_job}_{mname}_overlay.mp4")
                            except Exception:
                                pass
                except Exception:
                    pass

                # Note: metrics result file paths are intentionally not injected into the
                # top-level response payload (consumer expects metrics under 'metrics').

                # overwrite the job json on disk with enriched payload so upload picks it up
                try:
                    out_path = Path(dest_dir) / f"{job_id}.json"
                    # Only persist allowed top-level keys in the job JSON to match
                    # expected consumer schema (avoid injecting extraneous top-level fields).
                    allowed = ('frame_count', 'dimension', 'skeleton_rows', 'skeleton_columns', 'debug', 'metrics', 'stgcn_inference')
                    filtered = {k: response_payload[k] for k in allowed if k in response_payload}
                    _safe_write_json(out_path, filtered)
                except Exception:
                    traceback.print_exc()
        except Exception:
            traceback.print_exc()
        # --- Prepare and persist canonical skeleton2d.csv in dest_dir for MMACTION client ---
        try:
            # prefer df_2d wide conversion produced earlier by tidy_to_wide or recreate from df_2d
            try:
                from metric_algorithm.runner_utils import tidy_to_wide
                wide2 = tidy_to_wide(df_2d, dimension='2d', person_idx=0) if (isinstance(df_2d, pd.DataFrame) and not df_2d.empty) else (wide2 if 'wide2' in locals() else pd.DataFrame())
            except Exception:
                wide2 = wide2 if 'wide2' in locals() else pd.DataFrame()

            # Normalize column names/order to canonical skeleton2d.csv ordering
            if isinstance(wide2, pd.DataFrame) and not wide2.empty:
                # ensure columns are in the exact order as sample skeleton2d.csv
                COCO_ORDER = [
                    'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                ]
                cols = []
                for j in COCO_ORDER:
                    cols.extend([f"{j}_x", f"{j}_y", f"{j}_c"])
                # Build a canonical skeleton2d.csv with exact COCO columns (Nose..RAnkle + _x/_y/_c)
                try:
                    COCO_ORDER = [
                        'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                    ]
                    cols_canonical = []
                    for j in COCO_ORDER:
                        cols_canonical.extend([f"{j}_x", f"{j}_y", f"{j}_c"])

                    # Prepare lower-case lookup for available aliases in wide2
                    records = wide2.to_dict(orient='records') if not wide2.empty else []
                    ske_rows = []
                    for rec in records:
                        # build a lowercased key->value map to match aliases case-insensitively
                        lc_map = {str(k).lower(): v for k, v in rec.items()}

                        newr = {}
                        for j in COCO_ORDER:
                            # check common aliases for x
                            x_keys = [f"{j}__x", f"{j}_x", f"{j}_X", f"{j}__X"]
                            y_keys = [f"{j}__y", f"{j}_y", f"{j}_Y", f"{j}__Y"]
                            c_keys = [f"{j}__c", f"{j}_c", f"{j}_conf", f"{j}_score"]
                            # lowercase-map versions
                            x_val = None
                            y_val = None
                            c_val = None
                            for k in x_keys:
                                if k.lower() in lc_map:
                                    x_val = lc_map[k.lower()]
                                    break
                            for k in y_keys:
                                if k.lower() in lc_map:
                                    y_val = lc_map[k.lower()]
                                    break
                            for k in c_keys:
                                if k.lower() in lc_map:
                                    c_val = lc_map[k.lower()]
                                    break
                            # coerce to floats or NaN-like None
                            try:
                                newr[f"{j}_x"] = float(x_val) if x_val is not None else ''
                            except Exception:
                                newr[f"{j}_x"] = ''
                            try:
                                newr[f"{j}_y"] = float(y_val) if y_val is not None else ''
                            except Exception:
                                newr[f"{j}_y"] = ''
                            try:
                                newr[f"{j}_c"] = float(c_val) if c_val is not None else ''
                            except Exception:
                                newr[f"{j}_c"] = ''
                        ske_rows.append(newr)

                    ske_df = pd.DataFrame(ske_rows, columns=cols_canonical)
                    ske_path = Path(dest_dir) / f'{job_id}_skeleton2d.csv'
                    ske_df.to_csv(ske_path, index=False)
                    response_payload.setdefault('debug', {})['mmaction_input_csv'] = str(ske_path)
                    try:
                        mmaction_start = mmaction_client.start_mmaction_from_csv(ske_path, dest_dir, job_id, dimension, response_payload)
                        response_payload.setdefault('debug', {})['mmaction_start'] = mmaction_start
                        # If the client returned a thread handle, attempt to join it (with timeout)
                        try:
                            th = None
                            if isinstance(mmaction_start, dict):
                                th = mmaction_start.get('thread')
                            # also consider debug flags that indicate a later-started thread
                            if th and hasattr(th, 'join'):
                                try:
                                    # allow a short timeout to avoid blocking long-running inference
                                    join_timeout = float(os.environ.get('MMACTION_JOIN_SECONDS', '5'))
                                except Exception:
                                    join_timeout = 5.0
                                try:
                                    th.join(timeout=join_timeout)
                                except Exception:
                                    pass
                                # after join (or timeout), read response file and merge (prefixed name)
                                try:
                                    resp_path = Path(dest_dir) / f"{result_basename}_stgcn_response.json"
                                    if resp_path.exists():
                                        try:
                                            _merge_stgcn_resp(resp_path, response_payload)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    except Exception:
                        response_payload.setdefault('debug', {})['mmaction_start_error'] = True
                except Exception:
                    response_payload.setdefault('debug', {})['mmaction_skeleton_write_error'] = True
        except Exception:
            traceback.print_exc()

        # If metrics did not create an overlay mp4, create a minimal overlay from copied images
        try:
            # look for any mp4 in dest_dir
            mp4s = list(Path(dest_dir).glob(f"{job_id}*.mp4"))
            if not mp4s:
                img_dir_check = Path(dest_dir) / 'img'
                # prefer color_dir when available (color_dir may not be in this scope, try to detect)
                imgs = []
                try:
                    if 'color_dir' in locals() and Path(color_dir).exists():
                        imgs = sorted([p for p in Path(color_dir).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                        src_desc = 'color_dir'
                    else:
                        imgs = sorted([p for p in img_dir_check.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]) if img_dir_check.exists() else []
                        src_desc = 'dest_img_dir'
                    # dedupe while preserving order
                    seen = set()
                    uniq = []
                    for p in imgs:
                        if p.name in seen:
                            continue
                        seen.add(p.name)
                        uniq.append(p)
                    imgs = uniq
                except Exception:
                    imgs = []
                    src_desc = 'none'

                if imgs:
                    try:
                        from metric_algorithm.runner_utils import images_to_mp4
                        out_mp4 = Path(dest_dir) / f"{job_id}_overlay.mp4"
                        overlay_fps = fps if fps is not None else 30.0
                        created, used = images_to_mp4(imgs, out_mp4, fps=overlay_fps, resize=None, filter_rendered=True, write_debug=True)
                        if created:
                            try:
                                (Path(dest_dir) / 'overlay_debug.json').write_text(json.dumps({'overlay_source': src_desc, 'images_used': used}), encoding='utf-8')
                            except Exception:
                                pass
                    except Exception:
                        # ignore if helper missing or fails
                        pass
        except Exception:
            pass

        # --- Reorganize received_payloads layout: move CSVs -> csv/, MP4s -> mp4/
        try:
            csv_dir = Path(dest_dir) / 'csv'
            mp4_dir = Path(dest_dir) / 'mp4'
            csv_dir.mkdir(parents=True, exist_ok=True)
            mp4_dir.mkdir(parents=True, exist_ok=True)

            # Move CSV files from dest root into csv/
            for p in sorted(Path(dest_dir).glob('*.csv')):
                try:
                    target = csv_dir / p.name
                    if target.exists():
                        # avoid overwrite; keep existing
                        p.unlink()
                    else:
                        p.replace(target)
                except Exception:
                    # non-fatal
                    pass

            # Move MP4 files from dest root into mp4/
            for p in sorted(Path(dest_dir).glob('*.mp4')):
                try:
                    target = mp4_dir / p.name
                    if target.exists():
                        p.unlink()
                    else:
                        p.replace(target)
                except Exception:
                    pass

            # Update response_payload.metrics overlay paths to be relative into mp4/ if files were moved
            try:
                if 'metrics' in response_payload and isinstance(response_payload['metrics'], dict):
                    for mname, mval in response_payload['metrics'].items():
                        if isinstance(mval, dict) and 'overlay_mp4' in mval and mval.get('overlay_mp4'):
                            ov = mval.get('overlay_mp4')
                            try:
                                ovp = Path(ov)
                                if not ovp.exists():
                                    # maybe the file was moved to mp4/
                                    cand = mp4_dir / ovp.name
                                    if cand.exists():
                                        # set relative path so downstream readers resolve against dest_dir
                                        mval['overlay_mp4'] = str(Path('mp4') / cand.name)
                                else:
                                    # file still at root; move it into mp4/ (already attempted) and update path
                                    mval['overlay_mp4'] = str(Path('mp4') / ovp.name)
                            except Exception:
                                # leave as-is if anything goes wrong
                                pass
            except Exception:
                pass

            # rewrite job json with updated relative paths
            try:
                out_path = Path(dest_dir) / f"{result_basename}.json"
                if not out_path.exists():
                    out_path = Path(dest_dir) / f"{job_id}.json"
                _safe_write_json(out_path, response_payload)
            except Exception:
                pass
            # --- Enrich missing per-metric data from metric CSVs in csv/ ---
            try:
                # look for metric CSVs (prefixed or legacy) and populate frame_data/summary
                def _read_metrics_csv(candidate: Path):
                    try:
                        df = pd.read_csv(candidate)
                        # normalize frame column
                        if 'Frame' in df.columns and 'frame' not in df.columns:
                            df = df.rename(columns={'Frame': 'frame'})
                        if 'frame' in df.columns:
                            df = df.set_index('frame')
                        else:
                            df.index.name = 'frame'
                        frame_data = {}
                        for idx, row in df.iterrows():
                            try:
                                fkey = int(idx)
                            except Exception:
                                fkey = str(idx)
                            rowd = {}
                            for col, val in row.items():
                                if pd.isna(val):
                                    rowd[col] = None
                                else:
                                    try:
                                        if hasattr(val, 'item'):
                                            val = val.item()
                                    except Exception:
                                        pass
                                    rowd[col] = val
                            frame_data[str(fkey)] = rowd
                        return frame_data
                    except Exception:
                        return None

                for metric_name in ('com_speed', 'swing_speed'):
                    try:
                        # prefer prefixed csv: '<dimension>_<job_id>_<metric>_metrics.csv'
                        pref = None
                        for p in csv_dir.iterdir():
                            if p.is_file() and p.name.endswith(f"_{job_id}_{metric_name}_metrics.csv"):
                                pref = p
                                break
                        if pref is None:
                            # fallback legacy
                            legacy = csv_dir / f"{job_id}_{metric_name}_metrics.csv"
                            pref = legacy if legacy.exists() else None
                        if pref and pref.exists():
                            frame_data = _read_metrics_csv(pref)
                            if frame_data:
                                response_payload.setdefault('metrics', {}).setdefault(metric_name, {})['frame_data'] = frame_data
                                # if the metric has a separate summary JSON, try to attach it
                                summary_pref = Path(dest_dir) / f"{metric_name}_summary.json"
                                # also try metric-specific patterns
                                maybe_summary = Path(dest_dir) / f"{job_id}_{metric_name}_summary.json"
                                if summary_pref.exists():
                                    try:
                                        response_payload['metrics'][metric_name]['summary'] = json.loads(summary_pref.read_text(encoding='utf-8'))
                                    except Exception:
                                        pass
                                elif maybe_summary.exists():
                                    try:
                                        response_payload['metrics'][metric_name]['summary'] = json.loads(maybe_summary.read_text(encoding='utf-8'))
                                    except Exception:
                                        pass
                    except Exception:
                        pass
            except Exception:
                pass
            # --- Also try to read combined metric_result.json and merge per-metric data ---
            try:
                metrics_result_path = None
                for p in Path(dest_dir).iterdir():
                    if p.is_file() and p.name.endswith(f"_{job_id}_metric_result.json"):
                        metrics_result_path = p
                        break
                if metrics_result_path is None:
                    legacy_mr = Path(dest_dir) / f"{job_id}_metric_result.json"
                    if legacy_mr.exists():
                        metrics_result_path = legacy_mr
                if metrics_result_path and metrics_result_path.exists():
                    try:
                        mr = json.loads(metrics_result_path.read_text(encoding='utf-8'))
                        # mr may be structured with per-metric dicts
                        if isinstance(mr, dict):
                            for mname, mval in (mr.get('metrics') or mr).items():
                                try:
                                    if mname not in response_payload.setdefault('metrics', {}):
                                        # no existing entry: copy entire metric dict
                                        response_payload['metrics'][mname] = mval
                                        continue

                                    # merge into existing metric entry
                                    tgt = response_payload['metrics'][mname]
                                    if not isinstance(mval, dict):
                                        # non-dict value: only set if missing
                                        if mname not in response_payload['metrics']:
                                            response_payload['metrics'][mname] = mval
                                        continue

                                    # 1) summary: prefer metric_result's summary (overwrite or set)
                                    if 'summary' in mval and isinstance(mval.get('summary'), dict):
                                        try:
                                            tgt['summary'] = mval['summary']
                                        except Exception:
                                            pass

                                    # 2) frame-wise data: accept either 'frame_data' or convert 'metrics_data' -> 'frame_data'
                                    # If metric_result provides 'frame_data', merge it. If it provides 'metrics_data',
                                    # attempt to extract a primary timeseries (e.g., 'com_speed_timeseries') and convert
                                    try:
                                        # merge existing frame_data if present
                                        if 'frame_data' in mval and isinstance(mval['frame_data'], dict):
                                            tgt_fd = tgt.setdefault('frame_data', {})
                                            for fk, fv in mval['frame_data'].items():
                                                if fk not in tgt_fd:
                                                    tgt_fd[fk] = fv

                                        elif 'metrics_data' in mval and isinstance(mval['metrics_data'], dict):
                                            # pick the most likely timeseries subkey
                                            md = mval['metrics_data']
                                            timeseries_key = None
                                            for k in md.keys():
                                                if 'timeseries' in k.lower() or 'time_series' in k.lower():
                                                    timeseries_key = k
                                                    break
                                            if timeseries_key is None:
                                                # fallback to first key
                                                keys = list(md.keys())
                                                if keys:
                                                    timeseries_key = keys[0]
                                            if timeseries_key and isinstance(md.get(timeseries_key), dict):
                                                src_ts = md.get(timeseries_key)
                                                tgt_fd = tgt.setdefault('frame_data', {})
                                                for fk, fv in src_ts.items():
                                                    if fk not in tgt_fd:
                                                        tgt_fd[str(fk)] = fv
                                            # also preserve raw metrics_data for completeness
                                            if 'metrics_data' not in tgt:
                                                tgt['metrics_data'] = mval['metrics_data']
                                    except Exception:
                                        pass

                                    # 3) overlay fields: copy or set overlay_mp4/overlay_s3
                                    for ok in ('overlay_mp4', 'overlay_s3'):
                                        try:
                                            if ok in mval and mval.get(ok):
                                                tgt[ok] = mval.get(ok)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            # ensure metrics_path points to the file we read
                            # Do not inject metrics_path into final payload; keep metrics under 'metrics'
                    except Exception:
                        pass
            except Exception:
                pass
            # Verify/repair skeleton2d.csv in csv/ directory: ensure COCO _x/_y/_c columns exist
            try:
                ske_dir = csv_dir
                # Try to find skeleton2d.csv with or without job_id prefix
                ske_csv = ske_dir / f'{job_id}_skeleton2d.csv'
                if not ske_csv.exists():
                    ske_csv = ske_dir / 'skeleton2d.csv'
                if ske_csv.exists():
                    try:
                        missing = mmaction_client._validate_csv_matches_coco(ske_csv)
                        if missing:
                            # attempt to rebuild from df_2d if available
                            if 'df_2d' in locals() and isinstance(df_2d, pd.DataFrame) and not df_2d.empty:
                                from metric_algorithm.runner_utils import tidy_to_wide
                                try:
                                    wide2_recon = tidy_to_wide(df_2d, dimension='2d', person_idx=0)
                                except Exception:
                                    wide2_recon = pd.DataFrame()
                                if isinstance(wide2_recon, pd.DataFrame) and not wide2_recon.empty:
                                    # recreate canonical skeleton csv rows
                                    COCO_ORDER = [
                                        'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                                    ]
                                    cols_canonical = [f"{j}_x" for j in COCO_ORDER] + [f"{j}_y" for j in COCO_ORDER] + [f"{j}_c" for j in COCO_ORDER]
                                    # build ske_rows similar to earlier logic
                                    ske_rows = []
                                    for rec in wide2_recon.to_dict(orient='records'):
                                        lc_map = {str(k).lower(): v for k, v in rec.items()}
                                        newr = {}
                                        for j in COCO_ORDER:
                                            x_keys = [f"{j}__x", f"{j}_x", f"{j}_X", f"{j}__X"]
                                            y_keys = [f"{j}__y", f"{j}_y", f"{j}_Y", f"{j}__Y"]
                                            c_keys = [f"{j}__c", f"{j}_c", f"{j}_conf", f"{j}_score"]
                                            x_val = next((lc_map[k.lower()] for k in x_keys if k.lower() in lc_map), None)
                                            y_val = next((lc_map[k.lower()] for k in y_keys if k.lower() in lc_map), None)
                                            c_val = next((lc_map[k.lower()] for k in c_keys if k.lower() in lc_map), None)
                                            try:
                                                newr[f"{j}_x"] = float(x_val) if x_val is not None else ''
                                            except Exception:
                                                newr[f"{j}_x"] = ''
                                            try:
                                                newr[f"{j}_y"] = float(y_val) if y_val is not None else ''
                                            except Exception:
                                                newr[f"{j}_y"] = ''
                                            try:
                                                newr[f"{j}_c"] = float(c_val) if c_val is not None else ''
                                            except Exception:
                                                newr[f"{j}_c"] = ''
                                        ske_rows.append(newr)
                                    ske_df = pd.DataFrame(ske_rows, columns=[f"{j}_{s}" for j in COCO_ORDER for s in ('x','y','c')])
                                    ske_df.to_csv(ske_csv, index=False)
                                    response_payload.setdefault('debug', {})['mmaction_input_csv_rebuilt'] = str(ske_csv)
                                else:
                                    response_payload.setdefault('debug', {})['mmaction_input_rebuild_failed'] = True
                            else:
                                response_payload.setdefault('debug', {})['mmaction_input_missing_df2'] = True
                    except Exception:
                        response_payload.setdefault('debug', {})['mmaction_input_validation_error'] = True
                else:
                    # No skeleton file in csv/ — try to write one from df_2d if present
                    if 'df_2d' in locals() and isinstance(df_2d, pd.DataFrame) and not df_2d.empty:
                        try:
                            from metric_algorithm.runner_utils import tidy_to_wide
                            wide2_recon = tidy_to_wide(df_2d, dimension='2d', person_idx=0)
                        except Exception:
                            wide2_recon = pd.DataFrame()
                        if isinstance(wide2_recon, pd.DataFrame) and not wide2_recon.empty:
                            COCO_ORDER = [
                                'Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle'
                            ]
                            ske_rows = []
                            for rec in wide2_recon.to_dict(orient='records'):
                                lc_map = {str(k).lower(): v for k, v in rec.items()}
                                newr = {}
                                for j in COCO_ORDER:
                                    x_keys = [f"{j}__x", f"{j}_x", f"{j}_X", f"{j}__X"]
                                    y_keys = [f"{j}__y", f"{j}_y", f"{j}_Y", f"{j}__Y"]
                                    c_keys = [f"{j}__c", f"{j}_c", f"{j}_conf", f"{j}_score"]
                                    x_val = next((lc_map[k.lower()] for k in x_keys if k.lower() in lc_map), None)
                                    y_val = next((lc_map[k.lower()] for k in y_keys if k.lower() in lc_map), None)
                                    c_val = next((lc_map[k.lower()] for k in c_keys if k.lower() in lc_map), None)
                                    try:
                                        newr[f"{j}_x"] = float(x_val) if x_val is not None else ''
                                    except Exception:
                                        newr[f"{j}_x"] = ''
                                    try:
                                        newr[f"{j}_y"] = float(y_val) if y_val is not None else ''
                                    except Exception:
                                        newr[f"{j}_y"] = ''
                                    try:
                                        newr[f"{j}_c"] = float(c_val) if c_val is not None else ''
                                    except Exception:
                                        newr[f"{j}_c"] = ''
                                ske_rows.append(newr)
                            ske_df = pd.DataFrame(ske_rows, columns=[f"{j}_{s}" for j in COCO_ORDER for s in ('x','y','c')])
                            ske_csv_parent = ske_csv.parent if 'ske_csv' in locals() else ske_dir
                            ensure_dir = lambda p: p.mkdir(parents=True, exist_ok=True)
                            ensure_dir(ske_csv_parent)
                            ske_df.to_csv(ske_csv_parent / f'{job_id}_skeleton2d.csv', index=False)
                            response_payload.setdefault('debug', {})['mmaction_input_csv_written'] = str(ske_csv_parent / f'{job_id}_skeleton2d.csv')
            except Exception:
                # non-fatal
                pass
        except Exception:
            # non-fatal; continue
            pass

        # --- FINAL WAIT: conservatively wait for metrics + MMACTION/STGCN responses before writing final JSON ---
        try:
            try:
                mmaction_final_wait = int(os.environ.get('MMACTION_FINAL_WAIT_SECONDS', '60'))
            except Exception:
                mmaction_final_wait = 60
            try:
                metrics_final_wait = int(os.environ.get('METRICS_FINAL_WAIT_SECONDS', '60'))
            except Exception:
                metrics_final_wait = 60

            # candidate paths for STGCN response
            resp_pref = Path(dest_dir) / f"{result_basename}_stgcn_response.json"
            resp_legacy = Path(dest_dir) / f"{job_id}_stgcn_response.json"

            # metrics readiness: prefer combined metric_result file, otherwise per-metric artifacts
            metrics_ready = False
            stgcn_ready = False

            import time
            start = time.time()
            interval = 0.5

            def check_metrics_ready():
                # prefer combined metric_result json
                for p in Path(dest_dir).iterdir():
                    if p.is_file() and p.name.endswith(f"_{job_id}_metric_result.json"):
                        return True
                if (Path(dest_dir) / f"{job_id}_metric_result.json").exists():
                    return True
                # fallback: check that per-metric CSVs exist for known metrics
                csv_dir = Path(dest_dir) / 'csv'
                if not csv_dir.exists():
                    return False
                needed = ['com_speed', 'swing_speed']
                for m in needed:
                    found = False
                    for p in csv_dir.iterdir():
                        if p.is_file() and (p.name.endswith(f"_{job_id}_{m}_metrics.csv") or p.name.endswith(f"{m}_metrics.csv") or p.name.endswith(f"{job_id}_{m}_metrics.csv")):
                            found = True
                            break
                    if not found:
                        return False
                return True

            def check_stgcn_ready():
                if resp_pref.exists() or resp_legacy.exists():
                    return True
                return False

            # Wait loop: require both metrics_ready and stgcn_ready or timeouts
            while True:
                now = time.time()
                elapsed = now - start

                # update flags
                if not metrics_ready:
                    metrics_ready = check_metrics_ready()
                if not stgcn_ready:
                    stgcn_ready = check_stgcn_ready()

                # break conditions:
                # - both ready
                if metrics_ready and stgcn_ready:
                    break
                # - metric timeout exceeded and stgcn ready: allow stgcn to proceed
                if stgcn_ready and elapsed >= metrics_final_wait:
                    break
                # - mmaction timeout exceeded and metrics ready
                if metrics_ready and elapsed >= mmaction_final_wait:
                    break
                # - total timeout exceeded (max of both waits)
                if elapsed >= max(metrics_final_wait, mmaction_final_wait):
                    break

                time.sleep(interval)

            # merge STGCN response if present
            try:
                chosen = resp_pref if resp_pref.exists() else (resp_legacy if resp_legacy.exists() else None)
                if chosen:
                    try:
                        resp_txt = chosen.read_text(encoding='utf-8')
                        resp_obj = json.loads(resp_txt)
                    except Exception:
                        resp_obj = None
                    if resp_obj:
                        try:
                            _merge_stgcn_resp(chosen, response_payload)
                        except Exception:
                            response_payload['stgcn_inference'] = resp_obj
            except Exception:
                pass

            # merge metric_result if present (reuse existing logic above) - re-run merge block to gather late files
            try:
                # prefer combined metric_result file
                metrics_result_path = None
                for p in Path(dest_dir).iterdir():
                    if p.is_file() and p.name.endswith(f"_{job_id}_metric_result.json"):
                        metrics_result_path = p
                        break
                if metrics_result_path is None:
                    legacy_mr = Path(dest_dir) / f"{job_id}_metric_result.json"
                    if legacy_mr.exists():
                        metrics_result_path = legacy_mr
                if metrics_result_path and metrics_result_path.exists():
                    try:
                        mr = json.loads(metrics_result_path.read_text(encoding='utf-8'))
                        if isinstance(mr, dict):
                            for mname, mval in (mr.get('metrics') or mr).items():
                                try:
                                    # overwrite or set metric entries with mr content (prefer mr)
                                    response_payload.setdefault('metrics', {})[mname] = mval
                                except Exception:
                                    pass
                            response_payload['metrics_path'] = str(metrics_result_path)
                    except Exception:
                        pass
            except Exception:
                pass

            # Finally write the atomic final JSON (filter top-level keys to the allowed set)
            try:
                final_path = Path(dest_dir) / f"{result_basename}.json"
                if not final_path.exists():
                    final_path = Path(dest_dir) / f"{job_id}.json"
                allowed = ('frame_count', 'dimension', 'skeleton_rows', 'skeleton_columns', 'debug', 'metrics', 'stgcn_inference', 'overlay_mp4')
                filtered = {k: response_payload[k] for k in allowed if k in response_payload}
                _safe_write_json(final_path, filtered)
                # Ensure a canonical <job_id>.json exists for downstream upload/consumers
                try:
                    canonical = Path(dest_dir) / f"{job_id}.json"
                    if not canonical.exists():
                        _safe_write_json(canonical, filtered)
                except Exception:
                    pass
                # remove partial if final written
                try:
                    if partial_out_path.exists():
                        partial_out_path.unlink()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        return response_payload

        # cleanup tmp if desired (do not remove in case debugging needed)
        # shutil.rmtree(tmp_dir)
    except Exception as e:
        # Save error info to dest_dir for debugging
        try:
            err_path = Path(dest_dir) / 'result_error.txt'
            with err_path.open('w', encoding='utf-8') as ef:
                ef.write(traceback.format_exc())
        except Exception:
            pass
        return {'message': 'ERROR', 'detail': str(e)}


def upload_result_to_s3(dest_dir: Path, job_id: str, s3_key: Optional[str] = None, result_bucket: Optional[str] = None):
    """Upload only the canonical job JSON and MP4s under dest_dir/mp4/ to the result S3 bucket.

    Behavior:
    - Upload dest_dir/{job_id}.json (or raise if missing)
    - Upload files under dest_dir/mp4/*.mp4 only
    - When s3_key is provided and looks like '<user>/<dimension>/...': upload under
      '{user}/{dimension}/{job_id}/' preserving filenames (JSON at prefix root, MP4s under prefix/mp4/)
    - Otherwise upload under 'results/' with same structure (results/{job_id}.json and results/mp4/{fname})
    - Default bucket if not provided: 'golf-result-s3'
    """
    try:
        bucket = result_bucket or os.environ.get('S3_RESULT_BUCKET_NAME') or os.environ.get('RESULT_S3_BUCKET') or 'golf-result-s3'
        s3 = boto3.client('s3')

        target_prefix = None
        if s3_key:
            try:
                k = s3_key.lstrip('/')
                parts = k.split('/')
                if len(parts) >= 2:
                    # user/dimension/job_id
                    target_prefix = f"{parts[0]}/{parts[1]}/{job_id}"
            except Exception:
                target_prefix = None

        # canonical job json only - look for {dimension}_{job_id}.json first, fallback to {job_id}.json
        # (process_and_save saves as result_basename.json = "{dimension}_{job_id}.json")
        local_json = None
        dimension_prefix = None
        
        # Try to infer dimension from available files
        for dim in ['3d', '2d']:
            candidate = Path(dest_dir) / f"{dim}_{job_id}.json"
            if candidate.exists():
                local_json = candidate
                dimension_prefix = dim
                break
        
        # Fallback: check legacy naming
        if local_json is None:
            candidate = Path(dest_dir) / f"{job_id}.json"
            if candidate.exists():
                local_json = candidate
        
        if local_json is None or not local_json.exists():
            raise FileNotFoundError(f'Result file not found: {local_json} (checked {dimension_prefix or "legacy naming"} format)')

        uploaded = []
        if target_prefix:
            job_key = f"{target_prefix}/{local_json.name}"
        else:
            job_key = f"results/{local_json.name}"
        
        # 1. JSON 파일 업로드
        # ensure ContentType for JSON so consumers see the correct type
        try:
            s3.upload_file(str(local_json), bucket, job_key, ExtraArgs={'ContentType': 'application/json'})
        except TypeError:
            # botocore older versions may not accept ExtraArgs in upload_file signature in some envs; fallback
            s3.upload_file(str(local_json), bucket, job_key)
        uploaded.append({'local': str(local_json), 'bucket': bucket, 'key': job_key})

        # only mp4s under dest_dir/mp4/
        mp4_dir = Path(dest_dir) / 'mp4'
        if mp4_dir.exists() and mp4_dir.is_dir():
            for mp in sorted(mp4_dir.glob('*.mp4')):
                fname = mp.name
                if target_prefix:
                    key = f"{target_prefix}/mp4/{fname}"
                else:
                    key = f"results/mp4/{fname}"
                
                # 2. MP4 파일 업로드 (메타데이터 수정)
                s3.upload_file(
                    str(mp),
                    bucket,
                    key,
                    ExtraArgs={
                        'ContentType': 'video/mp4',         # 💡 브라우저 재생을 위한 Content-Type
                        'ContentDisposition': 'inline'      # 💡 다운로드가 아닌 인라인 표시(재생) 유도
                    }
                )
                uploaded.append({'local': str(mp), 'bucket': bucket, 'key': key})

        # Write a small manifest into dest_dir so operators can inspect what was uploaded
        try:
            manifest_path = Path(dest_dir) / 'uploaded_files.json'
            manifest_path.write_text(json.dumps({'uploaded': uploaded}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

        # Log to stdout as well for container logs
        try:
            print(f"[upload_result_to_s3] Uploaded files for job_id={job_id}: {uploaded}")
        except Exception:
            pass

        return {'message': 'UPLOADED', 'files': uploaded}
    except Exception as e:
        try:
            err_path = Path(dest_dir) / 'upload_error.txt'
            with err_path.open('w', encoding='utf-8') as ef:
                ef.write(traceback.format_exc())
        except Exception:
            pass
        return {'message': 'ERROR', 'detail': str(e)}