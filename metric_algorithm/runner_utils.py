"""Utilities to standardize metric modules' run_from_context interface.

Usage:
  - Import helpers and use the example template below to implement
    `run_from_context(ctx)` in metric modules.

Contract (recommended for run_from_context):
  - Accepts: ctx (dict) containing keys like 'wide2', 'wide3', 'dest_dir', 'job_id', 'img_dir', 'fps'
  - Returns: JSON-serializable dict (only ints/floats/strs/lists/dicts/None)
  - Writes per-metric CSV and overlay mp4 under dest_dir with job-prefixed names
  - If S3 bucket env var set, helper can upload overlay and return S3 info

This file provides small helpers to make that reliable and consistent.
"""
from pathlib import Path
import os
import json
import boto3
import numpy as _np
import pandas as _pd
import subprocess
import shutil
from typing import Any, Dict, Optional
# Provide tidy_to_wide here for backward-compatible imports from controller
try:
    from .utils_io import tidy_to_wide
except Exception:
    # If import fails, define a placeholder that will raise when used
    def tidy_to_wide(*args, **kwargs):
        raise ImportError('tidy_to_wide is not available from metric_algorithm.utils_io')


def normalize_value(v: Any):
    """Recursively convert numpy/pandas types to Python built-ins for JSON serialization.
    
    NaN, Inf, -Inf을 None으로 변환하여 프론트엔드 호환성 보장.
    """
    # numpy scalar
    if isinstance(v, (_np.generic,)):
        try:
            val = v.item()
            # Check for NaN/Inf
            if isinstance(val, float):
                if _np.isnan(val) or _np.isinf(val):
                    return None
            return val
        except Exception:
            try:
                val = float(v)
                if _np.isnan(val) or _np.isinf(val):
                    return None
                return val
            except Exception:
                return None

    # Check for NaN/Inf in float
    if isinstance(v, float):
        if _np.isnan(v) or _np.isinf(v):
            return None
        return v

    # numpy array -> list
    if isinstance(v, _np.ndarray):
        try:
            lst = v.tolist()
            # Recursively clean NaN/Inf from list
            return _clean_nan_inf_recursive(lst)
        except Exception:
            return None

    # pandas
    if isinstance(v, (_pd.Series,)):
        try:
            lst = v.tolist()
            return _clean_nan_inf_recursive(lst)
        except Exception:
            return None
    if isinstance(v, (_pd.DataFrame,)):
        # convert DataFrame to list-of-records (shallow)
        try:
            result = v.where(_pd.notnull(v), None).to_dict(orient="records")
            return _clean_nan_inf_recursive(result)
        except Exception:
            return None

    # dict/list/tuple
    if isinstance(v, dict):
        return {str(k): normalize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize_value(x) for x in v]

    # basic types
    if isinstance(v, (str, int, bool)) or v is None:
        return v

    # fallback: try to stringify
    try:
        return str(v)
    except Exception:
        return None


def _clean_nan_inf_recursive(obj):
    """Recursively replace NaN/Inf with None in nested structures."""
    if isinstance(obj, dict):
        return {k: _clean_nan_inf_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_nan_inf_recursive(item) for item in obj]
    elif isinstance(obj, float):
        if _np.isnan(obj) or _np.isinf(obj):
            return None
        return obj
    return obj


def _safe_num_to_python(x):
    """Convert numpy/float values to plain Python numbers or None for non-finite.

    This ensures JSON serialization produces `null` instead of `NaN`.
    프론트엔드 호환성: NaN/Inf는 None(null)으로 변환
    """
    try:
        xf = float(x)
    except Exception:
        return None
    return xf if _np.isfinite(xf) else None


def normalize_result(obj: Any) -> Any:
    """Normalize an arbitrary result to JSON-serializable structure.

    Typical usage: out['metrics'][name] = normalize_result(result)
    """
    return normalize_value(obj)


def ensure_dir(p: Path):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_valid_overlay_point(x, y, img_w: int, img_h: int, edge_margin: int = 2) -> bool:
    """Return True if (x,y) is a plausible overlay point inside image bounds.

    - Treat NaN or non-numeric as invalid.
    - Treat exact (0,0) as invalid (common sentinel from pose detectors).
    - Treat points within `edge_margin` pixels of the image edges as invalid
      since these often indicate occlusion/invalid detections for overlay.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        xf = float(x)
        yf = float(y)
    except Exception:
        return False
    # NaN
    try:
        if _np is not None and _np.isnan(xf) or _np is None and (xf != xf):
            return False
        if _np is not None and _np.isnan(yf) or _np is None and (yf != yf):
            return False
    except Exception:
        pass

    # sentinel at origin
    if xf == 0.0 and yf == 0.0:
        return False

    # image-edge guard
    try:
        if img_w is not None and img_h is not None:
            if xf <= edge_margin or yf <= edge_margin:
                return False
            if xf >= (img_w - 1 - edge_margin) or yf >= (img_h - 1 - edge_margin):
                return False
    except Exception:
        pass

    return True


def prepare_overlay_df(df, prefer_2d: bool = True, zero_threshold: float = 0.0):
    """Return a copy of `df` prepared for overlay rendering.

    - Does NOT modify the input `df` in-place.
    - Treats rows where both x and y are (near-)zero as missing and
      performs strong interpolation (linear) and forward/back-fill so that
      overlay drawing will not see (0,0) sentinel points.
    - Only intended for visualization; do NOT use this for metric computations.

    Args:
        df: pandas.DataFrame-like (must support columns access)
        prefer_2d: whether to prefer 2D-style column suffixes when searching
        zero_threshold: treat absolute values <= this as zero sentinel (0.0 by default)
    Returns:
        a new pandas.DataFrame with NaNs filled via interpolation + ffill/bfill
    """
    try:
        import pandas as _pd
        import numpy as _np
    except Exception:
        return df

    if df is None:
        return df
    try:
        df2 = df.copy(deep=True)
    except Exception:
        try:
            df2 = df.copy()
        except Exception:
            return df

    cols = list(df2.columns)
    axis_patterns = [('_x', '_y'), ('__x', '__y'), ('_X', '_Y'), ('_X3D', '_Y3D')] if prefer_2d else [('_X3D', '_Y3D'), ('__x', '__y'), ('_X', '_Y'), ('_x', '_y')]
    pairs = []
    col_set = set(cols)
    for c in cols:
        for xp, yp in axis_patterns:
            if c.endswith(xp):
                base = c[:-len(xp)]
                x_col = base + xp
                y_col = base + yp
                if x_col in col_set and y_col in col_set:
                    pairs.append((x_col, y_col))
                    break

    for x_col, y_col in pairs:
        try:
            sx = _pd.to_numeric(df2[x_col], errors='coerce')
            sy = _pd.to_numeric(df2[y_col], errors='coerce')
        except Exception:
            continue

        if zero_threshold is not None and zero_threshold >= 0.0:
            try:
                mask_both = (sx.abs() <= float(zero_threshold)) & (sy.abs() <= float(zero_threshold))
                sx = sx.mask(mask_both)
                sy = sy.mask(mask_both)
            except Exception:
                pass

        try:
            sx_interp = sx.interpolate(method='linear', limit_direction='both')
            sx_interp = sx_interp.ffill().bfill()
        except Exception:
            sx_interp = sx.ffill().bfill()
        try:
            sy_interp = sy.interpolate(method='linear', limit_direction='both')
            sy_interp = sy_interp.ffill().bfill()
        except Exception:
            sy_interp = sy.ffill().bfill()

        try:
            df2[x_col] = sx_interp
            df2[y_col] = sy_interp
        except Exception:
            try:
                df2.loc[:, x_col] = sx_interp
                df2.loc[:, y_col] = sy_interp
            except Exception:
                pass

    return df2


def write_df_csv(df: _pd.DataFrame, dest_dir: Path, job_id: str, metric: str, suffix: str = 'metrics.csv') -> str:
    """Write DataFrame to dest_dir/{job_id}_{metric}_{suffix} and return path string."""
    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)
    path = dest_dir / f"{job_id}_{metric}_{suffix}"
    # convert pandas NaN to empty for CSV (CSV will be read by metric CLIs too)
    df.to_csv(path, index=False)
    return str(path)


def relocate_overlay(local_overlay: str, dest_dir: Path, job_id: str, metric: str, target_name: Optional[str] = None) -> str:
    """Move or rename an existing overlay file into dest_dir with standardized job-prefixed name.

    If local_overlay already in dest_dir with the right name, returns it.
    """
    dest_dir = Path(dest_dir)
    ensure_dir(dest_dir)
    if target_name is None:
        target_name = f"{job_id}_{metric}_overlay.mp4"
    target = dest_dir / target_name
    src = Path(local_overlay)
    if not src.exists():
        raise FileNotFoundError(str(src))
    # if src and target are same, just return
    try:
        if src.resolve() == target.resolve():
            return str(target)
    except Exception:
        pass
    # copy/move - prefer rename to preserve speed when same filesystem
    try:
        src.replace(target)
    except Exception:
        # fallback to copy
        import shutil
        shutil.copy2(str(src), str(target))
    return str(target)


def upload_overlay_to_s3(local_path: str, job_id: str, metric: str, bucket_envs=('S3_RESULT_BUCKET_NAME', 'RESULT_S3_BUCKET')) -> Optional[Dict[str, str]]:
    """Upload overlay to S3 if bucket configured. Returns {'bucket','key'} or None.

    Key pattern: {job_id}_{metric}_overlay.mp4
    """
    bucket = None
    for env in bucket_envs:
        bucket = os.environ.get(env)
        if bucket:
            break
    if not bucket:
        return None

    path = Path(local_path)
    if not path.exists():
        return None

    key = f"{job_id}_{metric}_overlay.mp4"
    s3 = boto3.client('s3')
    # Upload (let boto3 raise if fails)
    s3.upload_file(str(path), bucket, key)
    return {'bucket': bucket, 'key': key}


def images_to_mp4(img_paths, out_mp4: str, fps: float = None, resize: tuple = None, filter_rendered: bool = True, write_debug: bool = True):
    """Create an mp4 from a list of image paths.

    - img_paths: iterable of Path or str
    - out_mp4: destination mp4 path (string)
    - fps: frames per second (default: 30.0 if None)
    - resize: (w,h) tuple to force resize, or None to use first image size
    - filter_rendered: if True skip files with 'render' or 'openpose' in the name
    - write_debug: if True, write a small JSON next to out_mp4 with source info
    Returns: (created: bool, used_count: int, original_fps: float, output_fps: float)
    """
    # Use provided fps, fall back to 30.0 if not specified
    if fps is None:
        fps = 30.0
    
    original_fps = float(fps)
    output_fps = float(fps)
    
    # Downsample to 60fps if original fps is > 60
    if output_fps > 60:
        output_fps = 60.0
        print(f"[INFO] Downsampling fps from {original_fps} to {output_fps}")
    
    # Debug: Check fps parameter
    print(f"[DEBUG] images_to_mp4: original_fps={original_fps}, output_fps={output_fps}")
    try:
        from pathlib import Path as _Path
        import cv2 as _cv2
        import json as _json
        paths = []
        for p in img_paths:
            pp = _Path(p)
            name = pp.name.lower()
            if filter_rendered and ('render' in name or 'openpose' in name):
                continue
            if not pp.exists():
                continue
            paths.append(pp)
        # dedupe by name preserving order
        seen = set()
        uniq = []
        for p in paths:
            if p.name in seen:
                continue
            seen.add(p.name)
            uniq.append(p)
        paths = uniq

        if not paths:
            return False, 0, original_fps, output_fps

        # determine size
        w, h = None, None
        if resize is not None:
            w, h = resize
        else:
            for p in paths:
                im = _cv2.imread(str(p))
                if im is None:
                    continue
                h, w = im.shape[:2]
                break
        if w is None or h is None:
            return False, 0, original_fps, output_fps

        fourcc = _cv2.VideoWriter_fourcc(*'mp4v')
        print(f"[DEBUG] cv2.VideoWriter call: output_fps={float(output_fps)}, size=({w}, {h})")
        vw = _cv2.VideoWriter(str(out_mp4), fourcc, float(output_fps), (w, h))
        
        # Time-based frame sampling for accurate downsampling
        # For 90fps → 60fps: we need to sample frames at 60fps intervals from 90fps source
        # Target frame interval in source time = 1 / 60 seconds = 1/60 seconds
        # Source frame time = 1 / 90 seconds
        # So we take frames at indices: 0, 1.5, 3, 4.5, 6, ... (in source frame units)
        
        target_frame_interval = original_fps / output_fps if output_fps > 0 else 1.0
        print(f"[DEBUG] Frame sampling: original_fps={original_fps}, output_fps={output_fps}, target_frame_interval={target_frame_interval:.2f}")
        
        used = 0
        frame_idx = 0
        next_frame_to_write = 0  # Next frame index (in source fps) to write to output
        
        for p in paths:
            im = _cv2.imread(str(p))
            if im is None:
                frame_idx += 1
                continue
            
            # Check if this frame should be written to output
            # We write frame when frame_idx >= next_frame_to_write
            if frame_idx >= next_frame_to_write:
                if im.shape[1] != w or im.shape[0] != h:
                    im = _cv2.resize(im, (w, h))
                vw.write(im)
                used += 1
                # Next frame to write is shifted by target_frame_interval
                next_frame_to_write += target_frame_interval
            frame_idx += 1
        vw.release()

        # Ensure MP4 is browser-friendly: transcode to H.264 (libx264) + faststart when ffmpeg is available.
        try:
            try:
                transcode_mp4_to_h264(str(out_mp4))
            except Exception:
                # fallback: try remux-only faststart
                try:
                    ensure_mp4_faststart(str(out_mp4))
                except Exception:
                    pass
        except Exception:
            pass

        if write_debug:
            try:
                dbg = {
                    'images_used': used, 
                    'total_candidates': len(paths), 
                    'original_fps': original_fps, 
                    'output_fps': output_fps,
                    'target_frame_interval': target_frame_interval,
                    'total_images_processed': frame_idx,
                    'sampling_ratio': used / frame_idx if frame_idx > 0 else 0
                }
                _Path(out_mp4).with_suffix('.debug.json').write_text(_json.dumps(dbg, indent=2), encoding='utf-8')
                print(f"[INFO] images_to_mp4 complete: used={used}/{frame_idx} frames ({100*used/frame_idx:.1f}%), fps={original_fps}→{output_fps}")
            except Exception:
                pass
        return True, used, original_fps, output_fps
    except Exception:
        return False, 0, original_fps if 'original_fps' in locals() else 30.0, output_fps if 'output_fps' in locals() else 30.0


def ensure_mp4_faststart(path: str) -> bool:
    """If `ffmpeg` is available, remux `path` in-place with `-movflags +faststart`.

    Returns True on success, False otherwise. This performs a copy remux (no re-encoding).
    The original file is replaced atomically when the operation succeeds.
    """
    try:
        if not path:
            return False
        p = Path(path)
        if not p.exists() or p.suffix.lower() != '.mp4':
            return False

        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            return False

        tmp = p.with_suffix('.faststart.tmp.mp4')
        # Run ffmpeg to remux without re-encoding
        cmd = [ffmpeg, '-y', '-i', str(p), '-c', 'copy', '-movflags', '+faststart', str(tmp)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

        if proc.returncode != 0 or not tmp.exists():
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

        # Replace original
        try:
            tmp.replace(p)
        except Exception:
            try:
                os.replace(str(tmp), str(p))
            except Exception:
                return False
        return True
    except Exception:
        return False


def transcode_mp4_to_h264(path: str, crf: int = 23, preset: str = 'fast', timeout: int = 300) -> bool:
    """Transcode (re-encode) an MP4 to H.264 (`libx264`) and apply `-movflags +faststart`.

    This is a safe, best-effort operation that produces a temporary file and
    atomically replaces the original on success. Returns True on success.
    """
    try:
        if not path:
            return False
        p = Path(path)
        p = Path(path)
        if not p.exists() or p.suffix.lower() != '.mp4':
            return False


        # use module-level is_valid_overlay_point if available

        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            return False

        tmp = p.with_suffix('.h264.tmp.mp4')
        # Use libx264 for video and copy audio if present. Apply faststart.
        cmd = [
            ffmpeg, '-y', '-i', str(p),
            '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
            '-c:a', 'copy',
            '-movflags', '+faststart', str(tmp)
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

        if proc.returncode != 0 or not tmp.exists():
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

        try:
            tmp.replace(p)
        except Exception:
            try:
                os.replace(str(tmp), str(p))
            except Exception:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 공통 유틸 함수들 (메트릭 모듈에서 중복 사용)
# ---------------------------------------------------------------------------

def parse_joint_axis_map_from_columns(columns, prefer_2d: bool = False) -> Dict[str, Dict[str, str]]:
    """관절명과 axis별 컬럼명을 매핑합니다.
    
    반환값 예시: {'Nose': {'x':'Nose__x','y':'Nose__y','z':'Nose__z'}, ...}
    
    지원 패턴:
      - Joint__x, Joint__y, Joint__z (더블 언더스코어, 우선)
      - Joint_X3D, Joint_Y3D, Joint_Z3D (3D 명시)
      - Joint_X, Joint_Y, Joint_Z (싱글 언더스코어)
      - Joint_x, Joint_y, Joint_z (소문자)
    """
    cols = list(columns)
    mapping: Dict[str, Dict[str, str]] = {}
    
    # 패턴 우선순위 (prefer_2d=False인 경우 3D 우선)
    if prefer_2d:
        axis_patterns = [
            ('_x', '_y', '_z'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_X3D', '_Y3D', '_Z3D'),
        ]
    else:
        axis_patterns = [
            ('_X3D', '_Y3D', '_Z3D'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_x', '_y', '_z'),
        ]
    
    col_set = set(cols)
    for col in cols:
        if col.lower() in ('frame', 'time', 'timestamp'):
            continue
        for x_pat, y_pat, z_pat in axis_patterns:
            if col.endswith(x_pat):
                joint = col[:-len(x_pat)]
                x_col = joint + x_pat
                y_col = joint + y_pat
                z_col = joint + z_pat
                if x_col in col_set and y_col in col_set:
                    mapping.setdefault(joint, {})['x'] = x_col
                    mapping.setdefault(joint, {})['y'] = y_col
                    if z_col in col_set:
                        mapping[joint]['z'] = z_col
                    break
    
    return mapping


def is_dataframe_3d(df: _pd.DataFrame) -> bool:
    """DataFrame이 3D(Z축 포함) 좌표를 가지는지 판단"""
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=False)
    # Z 좌표가 있는 관절 찾기
    for _, axes in cols_map.items():
        if 'x' in axes and 'y' in axes and 'z' in axes:
            return True
    # 추가 확인: Z 컬럼 직접 검색
    cols_lower = [c.lower() for c in df.columns]
    for c in cols_lower:
        if '__z' in c or '_z3d' in c or '_Z3D' in c:
            return True
    return False


def get_xyz_cols(df: _pd.DataFrame, name: str) -> Optional[_np.ndarray]:
    """지정한 관절의 3D 좌표(X,Y,Z) 추출 → (N,3) numpy array"""
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=False)
    if name in cols_map and all(a in cols_map[name] for a in ('x', 'y', 'z')):
        m = cols_map[name]
        try:
            arr = df[[m['x'], m['y'], m['z']]].astype(float).to_numpy()
            return arr
        except Exception:
            pass
    return None


def get_xy_cols_2d(df: _pd.DataFrame, name: str) -> Optional[_np.ndarray]:
    """지정한 관절의 2D 좌표(x,y) 추출 → (N,2) numpy array"""
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    if name in cols_map and all(a in cols_map[name] for a in ('x', 'y')):
        m = cols_map[name]
        try:
            arr = df[[m['x'], m['y']]].astype(float).to_numpy()
            return arr
        except Exception:
            pass
    return None


def get_xyc_row(row: _pd.Series, name: str, prefer_2d: bool = True):
    """한 행(row)에서 관절의 2D 좌표(x,y,c) 추출"""
    cols_map = parse_joint_axis_map_from_columns(row.index, prefer_2d=prefer_2d)
    x = y = c = _np.nan
    
    if name in cols_map:
        m = cols_map[name]
        try:
            x = float(row.get(m.get('x', ''), _np.nan))
        except Exception:
            pass
        try:
            y = float(row.get(m.get('y', ''), _np.nan))
        except Exception:
            pass
    
    # 신뢰도 추출 (다양한 컬럼명 지원)
    for c_name in [f"{name}__c", f"{name}_c", f"{name}_conf", f"{name}_score"]:
        if c_name in row.index:
            try:
                c_val = float(row.get(c_name, _np.nan))
                if not _np.isnan(c_val) and _np.isfinite(c_val):
                    c = c_val
                    break
            except Exception:
                pass
    
    if _np.isnan(c):
        c = 1.0  # 기본값
    
    return x, y, c


def scale_xy_for_overlay(x, y, img_w: int, img_h: int, data_range=None, margin: float = 0.1):
    """입력 좌표를 화면 좌표로 변환 (정규화 또는 픽셀)
    
    Args:
        x, y: 입력 좌표 (픽셀 또는 정규화)
        img_w, img_h: 화면 크기 (픽셀)
        data_range: (x_min, x_max, y_min, y_max) - 정규화 좌표 범위
        margin: 화면 여백 비율 (기본 0.1 = 10%)
    
    Returns:
        (screen_x, screen_y) 또는 (NaN, NaN)
    """
    try:
        if _np.isnan(x) or _np.isnan(y):
            return _np.nan, _np.nan
        xf = float(x)
        yf = float(y)
    except Exception:
        return _np.nan, _np.nan
    
    # 작은 범위(정규화) 좌표면 화면으로 매핑
    if data_range is not None:
        x_min, x_max, y_min, y_max = data_range
        dx = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
        dy = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
        x_norm = (xf - x_min) / dx
        y_norm = (yf - y_min) / dy
        screen_x = (margin + x_norm * (1 - 2*margin)) * img_w
        screen_y = (margin + y_norm * (1 - 2*margin)) * img_h
        return screen_x, screen_y
    
    # 그렇지 않으면 픽셀 좌표로 간주
    return xf, yf


def compute_overlay_range(df: _pd.DataFrame, kp_names: list, prefer_2d: bool = True):
    """DataFrame의 관절 좌표 범위 계산 → (x_min, x_max, y_min, y_max, is_small_range)"""
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=prefer_2d)
    xs, ys = [], []
    
    for name in kp_names:
        ax = cols_map.get(name, {})
        cx = ax.get('x')
        cy = ax.get('y')
        
        if cx in df.columns:
            try:
                vals = _pd.to_numeric(df[cx], errors='coerce').dropna()
                xs.extend(vals.tolist())
            except Exception:
                pass
        
        if cy in df.columns:
            try:
                vals = _pd.to_numeric(df[cy], errors='coerce').dropna()
                ys.extend(vals.tolist())
            except Exception:
                pass
    
    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # 값이 작으면 정규화된 좌표로 판단
        is_small = all(abs(v) <= 2.0 for v in (x_min, x_max, y_min, y_max))
        return x_min, x_max, y_min, y_max, is_small
    
    return None, None, None, None, False


# ---------------------------------------------------------------------------
# Example template for run_from_context (copy into metric modules):
#
# from metric_algorithm.runner_utils import (
#     normalize_result, write_df_csv, relocate_overlay, upload_overlay_to_s3, ensure_dir,
#     parse_joint_axis_map_from_columns, is_dataframe_3d, get_xyz_cols, get_xy_cols_2d
# )
#
# def run_from_context(ctx: dict):
#     dest = Path(ctx.get('dest_dir', '.'))
#     job_id = str(ctx.get('job_id', 'job'))
#     wide3 = ctx.get('wide3')
#     wide2 = ctx.get('wide2')
#     fps = int(ctx.get('fps', 30))
#     metric = 'my_metric'
#     ensure_dir(dest)
#     out = {}
#     try:
#         df = wide3 if wide3 is not None else wide2
#         if df is None:
#             return {'error': 'No DataFrame provided', 'metrics_csv': None}
#         
#         dim = '3d' if is_dataframe_3d(df) else '2d'
#         
#         # perform metric computation using standard functions
#         # e.g., df_metrics = compute_metric_df(df)
#         metrics_csv = write_df_csv(df_metrics, dest, job_id, metric)
#         out['metrics_csv'] = metrics_csv
#         out['dimension'] = dim
#         out['summary'] = normalize_result({'mean': df_metrics['value'].mean()})
#     except Exception as e:
#         return {'error': str(e), 'metrics_csv': None}
#
#     # overlay (if available)
#     try:
#         if wide2 is not None:
#             # generate overlay mp4
#             overlay_local = str(dest / f"{job_id}_{metric}_overlay.mp4")
#             # call overlay drawing function
#             out['overlay_mp4'] = overlay_local
#             s3info = upload_overlay_to_s3(overlay_local, job_id, metric)
#             if s3info:
#                 out['overlay_s3'] = s3info
#     except Exception as e:
#         out.setdefault('overlay_error', str(e))
#
#     return normalize_result(out)
#
# ---------------------------------------------------------------------------
