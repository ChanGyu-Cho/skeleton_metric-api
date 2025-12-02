# src/swing_speed.py
# -*- coding: utf-8 -*-
"""
Swing Speed 전용 분석기
- 양쪽 손목(LWrist, RWrist) 관절만 시각화
- Grip 포인트와 Swing Speed 계산 및 시각적 피드백
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import glob
import re
import math
from typing import Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import yaml
except ImportError:
    yaml = None

# 공통 유틸리티 임포트
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from utils_io import natural_key, ensure_dir
from impact_utils import detect_impact_by_crossing, compute_stance_mid_and_width as compute_stance_utils
from runner_utils import (
    parse_joint_axis_map_from_columns,
    is_dataframe_3d,
    get_xyz_cols,
    get_xy_cols_2d,
    get_xyc_row,
    scale_xy_for_overlay,
    compute_overlay_range,
    write_df_csv,
    images_to_mp4,
    upload_overlay_to_s3,
    normalize_result,
)

# 7번 아이언 길이 (37 inch ≒ 0.94 m) 고정 가정
CLUB_LENGTH_7I_M = 0.94
EPS_R = 1e-6  # 반지름 0 방지용

# =========================================================
# 2D 스무딩 유틸들 (점프 제한 없는 필터들)
# =========================================================
def _interpolate_series(s: pd.Series) -> pd.Series:
    if s.isna().all():
        return s.copy()
    s2 = s.copy()
    s2 = s2.astype(float)
    s2 = s2.interpolate(method='linear', limit_direction='both')
    s2 = s2.ffill().bfill()
    return s2


def suppress_jumps(arr, k: float = 5.0):
    """
    Suppress momentary large jumps in a 1D coordinate sequence using MAD-based thresholding.
    Replaces values that jump beyond median+ k*MAD by a limited increment from previous value.
    """
    arr = np.asarray(arr, dtype=float)
    out = arr.copy()
    if len(arr) <= 1:
        return out

    deltas = np.diff(arr, prepend=arr[0])
    abs_deltas = np.abs(deltas)

    med = np.median(abs_deltas)
    mad = np.median(np.abs(abs_deltas - med))
    thresh = med + k * 1.4826 * mad

    for i in range(1, len(arr)):
        if abs_deltas[i] > thresh:
            # limit the step to threshold in the same sign direction
            out[i] = out[i-1] + np.sign(deltas[i]) * thresh
    return out

def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(arr, dtype=float)
    y[:] = np.nan
    prev = None
    for i, v in enumerate(arr):
        if np.isnan(v):
            y[i] = prev if prev is not None else np.nan
            continue
        prev = v if prev is None else (alpha * v + (1 - alpha) * prev)
        y[i] = prev
    return pd.Series(y).ffill().bfill().to_numpy()

def _moving(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().to_numpy()

def _median(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()

def _gaussian(arr: np.ndarray, window: int, sigma: Optional[float]) -> np.ndarray:
    if window <= 1:
        return arr
    if sigma is None:
        sigma = max(1.0, window / 3.0)
    # 가우시안 커널 생성
    half = window // 2
    xs = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    # NaN 보간 후 컨볼브
    s = _interpolate_series(pd.Series(arr))
    y = np.convolve(s.to_numpy(), kernel, mode='same')
    return y

def _hampel(arr: np.ndarray, window: int, n_sigmas: float, alpha: float) -> np.ndarray:
    if window <= 1:
        return arr
    s = pd.Series(arr)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    # 1.4826 * MAD ≈ 표준편차 추정치
    thresh = n_sigmas * 1.4826 * mad
    out = s.copy()
    mask = (s - med).abs() > thresh
    out[mask] = med[mask]
    return _ema(out.to_numpy(), alpha)

def _one_euro(arr: np.ndarray, fps: int, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
    # https://cristal.univ-lille.fr/~casiez/1euro/
    if fps is None or fps <= 0:
        fps = 30
    dt = 1.0 / float(fps)
    def alpha(fc):
        tau = 1.0 / (2 * math.pi * fc)
        return 1.0 / (1.0 + tau / dt)
    prev_x = None
    prev_dx = 0.0
    xhat = []
    for x in arr:
        if np.isnan(x):
            xhat.append(prev_x if prev_x is not None else np.nan)
            continue
        # 미분 추정
        dx = 0.0 if prev_x is None else (x - prev_x)
        ad = alpha(d_cutoff)
        dx_hat = ad * dx + (1 - ad) * prev_dx
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = alpha(cutoff)
        x_f = x if prev_x is None else (a * x + (1 - a) * prev_x)
        prev_x, prev_dx = x_f, dx_hat
        xhat.append(x_f)
    return pd.Series(xhat).ffill().bfill().to_numpy()

def smooth_df_2d(
    df: pd.DataFrame,
    prefer_2d: bool,
    method: str = 'ema',
    window: int = 5,
    alpha: float = 0.2,
    fps: Optional[int] = None,
    gaussian_sigma: Optional[float] = None,
    hampel_sigma: float = 3.0,
    oneeuro_min_cutoff: float = 1.0,
    oneeuro_beta: float = 0.007,
    oneeuro_d_cutoff: float = 1.0,
) -> pd.DataFrame:
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=prefer_2d)
    out = df.copy()
    for joint, axes in cols_map.items():
        for ax in ('x', 'y'):
            col = axes.get(ax)
            if not col or col not in out.columns:
                continue
            s = out[col].astype(float)
            s_interp = _interpolate_series(s)
            arr = s_interp.to_numpy()
            # suppress single-frame spikes before smoothing
            arr = suppress_jumps(arr, k=5.0)
            if method == 'ema':
                y = _ema(arr, alpha)
            elif method == 'moving':
                y = _moving(arr, window)
            elif method == 'median':
                y = _median(arr, window)
            elif method == 'gaussian':
                y = _gaussian(arr, window, gaussian_sigma)
            elif method == 'hampel_ema':
                y = _hampel(arr, window, hampel_sigma, alpha)
            elif method == 'oneeuro':
                y = _one_euro(arr, fps, oneeuro_min_cutoff, oneeuro_beta, oneeuro_d_cutoff)
            else:
                y = arr
            # 원래 NaN은 유지
            y_series = pd.Series(y, index=s.index)
            y_series[s.isna()] = np.nan
            out[col] = y_series
    print(f"🎛️ 2D 스무딩 적용: method={method}, window={window}, alpha={alpha}")
    return out

def _filter_depth_outliers(z_coords: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Z 좌표의 이상치를 필터링합니다 (depth 추적 오류 제거).
    
    방법: IQR (Interquartile Range) 기반 이상치 감지 (더 보수적)
    - Q1/Q3 계산
    - IQR = Q3 - Q1
    - 이상치 범위: [Q1 - 3.0*IQR, Q3 + 3.0*IQR]  ← 기존 1.5에서 3.0으로 완화 (더 극단적인 이상치만 제거)
    - 범위 밖의 값 → 보간으로 대체
    
    예시: Z 좌표가 [2200, 2300, ..., 15000 (이상!), ..., 2400]
          매우 극단적인 이상치만 제거하고 보간
    """
    z = z_coords.astype(float).copy()
    
    # 유효한 값 필터링
    valid_mask = np.isfinite(z)
    if not np.any(valid_mask):
        return z
    
    valid_vals = z[valid_mask]
    q1 = np.percentile(valid_vals, 25)
    q3 = np.percentile(valid_vals, 75)
    iqr = q3 - q1
    
    # 이상치 범위 정의 (3.0 * IQR로 완화 - 매우 극단적인 경우만 제거)
    lower_bound = q1 - 3.0 * iqr
    upper_bound = q3 + 3.0 * iqr
    
    # 이상치 마스크
    outlier_mask = (z < lower_bound) | (z > upper_bound)
    
    if verbose:
        n_outliers = np.sum(outlier_mask)
        if n_outliers > 0:
            outlier_indices = np.where(outlier_mask)[0]
            print(f"[DEBUG] 좌표 이상치 감지: {n_outliers}개")
            print(f"[DEBUG] IQR 범위: [{lower_bound:.1f}, {upper_bound:.1f}]")
            print(f"[DEBUG] 이상치 프레임: {list(outlier_indices[:5])}{'...' if n_outliers > 5 else ''}")
            print(f"[DEBUG] 이상치 값: {z[outlier_mask][:5]}")
    
    # 이상치를 NaN으로 마킹하여 보간
    z[outlier_mask] = np.nan
    
    # 보간으로 NaN 채우기
    z_series = pd.Series(z)
    z_filtered = z_series.interpolate(method='linear', limit_direction='both').ffill().bfill().to_numpy()
    
    if verbose and np.sum(outlier_mask) > 0:
        print(f"[DEBUG] 이상치를 보간으로 대체")
    
    return z_filtered


def vectorized_speed_m_s_3d(points_xyz: np.ndarray, fps: int, scale_to_m: float = 1.0, 
                            filter_z_outliers: bool = True) -> np.ndarray:
    """
    벡터화된 손목 3D 속도(m/s) 계산
    
    공식: v = Δs / Δt [m/s]
    - Δs = sqrt((Δx)^2 + (Δy)^2 + (Δz)^2) : 연속 프레임 간 유클리드 거리 (미터 단위)
    - Δt = 1 / fps : 프레임 간격 (초)
    - scale_to_m: 좌표 단위를 미터로 환산하는 스케일
      * 카메라 정규화 좌표: 1.0 (0.0002~0.0005 범위, 이미 m 단위)
      * mm 좌표: 0.001 (mm → m)
      * cm 좌표: 0.01 (cm → m)
    - filter_z_outliers: True이면 depth 이상치(Z 좌표) + XY 좌표 필터링
    
    정확한 계산 순서:
    1. XYZ 이상치 필터링 (depth + 2D 추적 오류 처리)
    2. Δs = sqrt((Δx)^2 + (Δy)^2 + (Δz)^2) [좌표 원래 단위]
    3. Δs_m = Δs * scale_to_m [m 단위로 변환]
    4. v_m_s = Δs_m * fps [m/s]
    
    반환: v_m_s 배열 (첫 번째 프레임은 0.0)
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        return np.full((len(points_xyz),), np.nan, dtype=float)
    X = points_xyz.astype(float).copy()
    
    # ⚠️ CRITICAL: XYZ 좌표 이상치 필터링 (depth 카메라 추적 오류 + 2D 추적 오류 제거)
    if filter_z_outliers:
        print(f"[DEBUG] XYZ 좌표 이상치 필터링 시작...")
        
        # 각 축별로 독립적으로 필터링
        for axis in range(3):
            axis_name = ['X', 'Y', 'Z'][axis]
            X[:, axis] = _filter_depth_outliers(X[:, axis], verbose=False)
            outlier_count = np.sum(~np.isfinite(X[:, axis]))
            if outlier_count > 0:
                print(f"[DEBUG] {axis_name} 축 이상치 처리 완료")
    
    # 각 축별 보간 (결측치 채우기)
    for c in range(3):
        s = pd.Series(X[:, c])
        s = s.interpolate(limit_direction='both').ffill().bfill()
        X[:, c] = s.to_numpy()
    
    # 프레임 간 거리 계산
    dx = np.diff(X[:, 0], prepend=X[0, 0])
    dy = np.diff(X[:, 1], prepend=X[0, 1])
    dz = np.diff(X[:, 2], prepend=X[0, 2])
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # 1단계: 좌표 단위를 m로 환산
    ds_m = ds * float(scale_to_m)
    
    # 2단계: v = Δs_m * fps [m/s]
    # (Δt = 1/fps이므로, Δs/Δt = Δs * fps)
    fps_float = float(fps if fps and fps > 0 else 30)
    v_m_s = ds_m * fps_float
    
    if len(v_m_s) > 0:
        v_m_s[0] = 0.0
    return v_m_s

def _speed_conversions_m_s(v_m_s: np.ndarray):
    """m/s 배열을 km/h, mph로 동시 변환
    
    공식:
    - v_km_h = v_m_s * 3.6 [m/s * 3600/1000 = m/s * 3.6 → km/h]
    - v_mph = v_m_s * 2.237094 [정확한 값: 1 m/s = 3.6 / 1.609344 mph]
    """
    KM_H_PER_M_S = 3.6
    MPH_PER_M_S = 3.6 / 1.609344  # 정확한 값: 2.237094...
    v_kmh = v_m_s * KM_H_PER_M_S
    v_mph = v_m_s * MPH_PER_M_S
    return v_m_s, v_kmh, v_mph



def is_dataframe_3d(df: pd.DataFrame) -> bool:
    """데이터프레임에 Z 축 좌표가 존재하는지 검사하여 3D 여부 판정"""
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=False)
    for axes in cols_map.values():
        if 'z' in axes:
            return True
    return False

def get_xy_cols_2d(df: pd.DataFrame, name: str) -> np.ndarray:
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    if name in cols_map and all(a in cols_map[name] for a in ('x','y')):
        m = cols_map[name]
        arr = df[[m['x'], m['y']]].astype(float).to_numpy()
        return arr
    return np.full((len(df), 2), np.nan, dtype=float)

def speed_2d(points_xy: np.ndarray, fps: Optional[int]):
    """2D 속도 계산(px/초 또는 px/프레임)"""
    N = len(points_xy)
    v = np.full(N, np.nan, dtype=float)
    for i in range(1, N):
        a, b = points_xy[i-1], points_xy[i]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            continue
        v[i] = float(np.linalg.norm(b - a))
    unit = "px/frame"
    if fps and fps > 0:
        v = v * float(fps)
        unit = "px/s"
    v = pd.Series(v).ffill().fillna(0).to_numpy()
    return v, unit


def run_from_context(ctx: dict):
    """Runner hook for controller: write metric CSVs and create overlay mp4 for swing_speed.

    Minimal implementation: if `wide2`/`wide3` exist, write CSVs and attempt to
    build an overlay mp4 from `ctx['img_dir']` or `dest_dir/img` and return
    `overlay_mp4`. Upload to S3 if configured.
    
    참고: write_df_csv, images_to_mp4, upload_overlay_to_s3, normalize_result는
    runner_utils에서 모듈 최상단에서 임포트됨
    """
    out = {}
    try:
        dest = Path(ctx.get('dest_dir') or ctx.get('dest') or '.')
        job_id = str(ctx.get('job_id') or ctx.get('job') or 'job')
        metric = 'swing_speed'
        wide2 = ctx.get('wide2')
        wide3 = ctx.get('wide3')
        try:
            if wide2 is not None:
                out['metrics_csv'] = write_df_csv(wide2, dest, job_id, metric)
            elif wide3 is not None:
                out['metrics_csv'] = write_df_csv(wide3, dest, job_id, metric)
        except Exception:
            pass

        try:
            img_dir = Path(ctx.get('img_dir') or (Path(dest) / 'img'))
            imgs = []
            if img_dir.exists() and img_dir.is_dir():
                imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if not imgs:
                op_dir = Path(dest) / 'openpose_img'
                if op_dir.exists() and op_dir.is_dir():
                    imgs = sorted([p for p in op_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if imgs:
                out_mp4 = Path(dest) / f"{job_id}_{metric}_overlay.mp4"
                # CRITICAL: Get fps from context; do NOT default to 30
                # fps must be provided by controller based on video/intrinsics metadata
                fps_ctx = ctx.get('fps')
                if fps_ctx is None:
                    print(f"[WARN] swing_speed: fps not provided in context, using fallback 30")
                    fps_value = 30
                else:
                    fps_value = fps_ctx
                print(f"[DEBUG] swing_speed: ctx.get('fps') = {fps_ctx}")
                print(f"[DEBUG] swing_speed: float(fps_value) = {float(fps_value)}")
                created, used, original_fps, output_fps = images_to_mp4(imgs, out_mp4, fps=float(fps_value), resize=None, filter_rendered=True, write_debug=True)
                if created:
                    out['overlay_mp4'] = str(out_mp4)
                    # Store fps information for frontend
                    out['fps_info'] = {
                        'original_fps': original_fps,
                        'output_fps': output_fps
                    }
                    try:
                        s3info = upload_overlay_to_s3(str(out_mp4), job_id, metric)
                        if s3info:
                            out['overlay_s3'] = s3info
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception as e:
        out['error'] = str(e)
    return normalize_result(out)

def _pair_distance_px_series_2d(df: pd.DataFrame, joint_a: str, joint_b: str) -> np.ndarray:
    """2D에서 두 관절 사이의 프레임별 거리(px) 시계열을 계산(보간/ffill/bfill 포함)."""
    A = get_xy_cols_2d(df, joint_a)
    B = get_xy_cols_2d(df, joint_b)
    # 보간
    for arr in (A, B):
        for c in range(arr.shape[1]):
            s = pd.Series(arr[:, c])
            s = s.interpolate(limit_direction='both').ffill().bfill()
            arr[:, c] = s.to_numpy()
    d = np.sqrt((A[:, 0] - B[:, 0])**2 + (A[:, 1] - B[:, 1])**2)
    return d

def _get_m_per_px_from_cfg(cfg: dict, df_overlay: pd.DataFrame) -> Optional[float]:
    """
    analyze.yaml에서 2D 보정 스케일(m/px)을 가져오거나, 관절 쌍 캘리브레이션으로 추정.
    지원 키:
      - calibration_2d:
          method: "joint_pair"
          joint_a: "LShoulder"
          joint_b: "RShoulder"
          real_length_m: 0.40
      - m_per_px_2d: 숫자 (예: 0.0025)
    
    우선순위 (정확도 순서):
      1. 수동 관절 쌍 캘리브레이션 (가장 정확함 - 실시간 측정)
      2. 자동 캘리브레이션 (신체 정보 기반)
      3. 직접 지정 (사전 계산값, fallback)
    
    반환: m_per_px 또는 None
    """
    # 1순위: 수동 관절 쌍 캘리브레이션 (가장 정확)
    calib = cfg.get("calibration_2d") or {}
    if isinstance(calib, dict) and calib.get("method", "").lower() == "joint_pair":
        ja = calib.get("joint_a")
        jb = calib.get("joint_b")
        rl = calib.get("real_length_m")
        if ja and jb and rl is not None:
            try:
                real_len_m = float(rl)
                if real_len_m <= 0:
                    raise ValueError
            except Exception:
                print("⚠️ calibration_2d.real_length_m 값이 유효하지 않습니다.")
            else:
                d_px = _pair_distance_px_series_2d(df_overlay, ja, jb)
                d_px_valid = d_px[np.isfinite(d_px) & (d_px > 0)]
                if d_px_valid.size > 0:
                    # 중앙값 사용(노이즈/자세 변화 완화)
                    px_med = float(np.median(d_px_valid))
                    m_per_px = real_len_m / px_med
                    print(f"🧭 [우선순위 1] 수동 관절 쌍 캘리브레이션: {ja}-{jb} median={px_med:.2f} px, real={real_len_m:.3f} m → m_per_px={m_per_px:.6f}")
                    return m_per_px
                else:
                    print("⚠️ 캘리브레이션용 관절 쌍 거리(px)를 계산할 수 없습니다.")
    
    # 2순위: 자동 캘리브레이션 (신체 정보 기반)
    auto_flag = True if calib.get("method", "").lower() in ("", "auto") else False
    if auto_flag:
        mpp_auto = _autocalibrate_m_per_px(df_overlay, cfg)
        if mpp_auto is not None:
            return mpp_auto
    
    # 3순위: 직접 지정 (fallback)
    mpp = cfg.get("m_per_px_2d")
    if mpp is not None:
        try:
            val = float(mpp)
            if val > 0:
                print(f"🧭 [우선순위 3] 2D 보정 스케일 직접 지정: m_per_px={val:.6f}")
                return val
        except Exception:
            pass
    
    return None

def _autocalibrate_m_per_px(df: pd.DataFrame, cfg: dict) -> Optional[float]:
    """
    피사체 신체 비율 기반의 자동 캘리브레이션.
    - 후보 관절쌍 중 프레임 내 중앙값 픽셀거리가 크고(해상도 유리), 변동률이 낮은(원근/자세 영향 적은) 쌍을 선택.
    - 실제 길이는 아래 우선순위를 사용:
        1) subject.shoulder_width_m
        2) subject.height_m * 0.259 (어깨폭 근사 비율)
        3) 기본값 0.40 m
    반환: m_per_px 또는 None
    """
    candidates = [
        ("LShoulder", "RShoulder", "shoulder"),
        ("LHip", "RHip", "hip"),
        ("LAnkle", "RAnkle", "ankle")
    ]
    stats = []
    for a, b, tag in candidates:
        d = _pair_distance_px_series_2d(df, a, b)
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size == 0:
            continue
        med = float(np.median(valid))
        # 변동률(CV) 계산 (중앙값 사용)
        mad = float(np.median(np.abs(valid - med))) if valid.size > 0 else 0.0
        cv = (mad / med) if med > 1e-6 else 1e9
        stats.append((a, b, tag, med, cv))
    if not stats:
        return None
    # 큰 길이(안정) + 낮은 변동률 선호: med/ cv 조합으로 정렬
    stats.sort(key=lambda x: (-x[3], x[4]))
    a, b, tag, px_med, cv = stats[0]

    subj = cfg.get("subject") or {}
    shoulder_w_m = subj.get("shoulder_width_m")
    height_m = subj.get("height_m")
    real_len_m = None
    if shoulder_w_m is not None:
        try:
            real_len_m = float(shoulder_w_m)
        except Exception:
            real_len_m = None
    if real_len_m is None and height_m is not None:
        try:
            h = float(height_m)
            if h > 0:
                real_len_m = 0.259 * h  # 어깨폭 근사 비율
        except Exception:
            pass
    if real_len_m is None:
        real_len_m = 0.40  # 기본 어깨폭

    m_per_px = real_len_m / px_med if px_med > 0 else None
    if m_per_px is not None:
        print(f"🧭 2D 자동 보정: pair={a}-{b} median={px_med:.2f}px, real≈{real_len_m:.3f}m → m_per_px={m_per_px:.6f} (cv={cv:.3f})")
    return m_per_px

def analyze_wrist_speed_3d(df: pd.DataFrame, fps: int, wrist: str = "RWrist", scale_to_m: float = 1.0):
    """
    입력: 3D CSV (좌표 원래 단위), 필수: {wrist}_X3D/Y3D/Z3D, RAnkle_X3D, LAnkle_X3D
    출력:
      - impact_frame, peak_frame
      - 시계열 속도 v_m_s, v_km_h, v_mph (모두 m/s 기준으로 변환됨)
      - 피크 속도(손목) km/h, mph
      - 7번 아이언(37") 각속도 기반 클럽 헤드 속도 추정
    
    계산 프로세스:
    1. 손목 3D 좌표(X, Y, Z) 추출
    2. 프레임 간 거리 계산 Δs (좌표 원래 단위)
    3. scale_to_m 적용하여 m 단위로 변환: Δs_m = Δs * scale_to_m
    4. 속도 계산: v_m_s = Δs_m * fps [m/s]
    5. 변환: v_km_h = v_m_s * 3.6, v_mph = v_m_s * 2.237
    6. 어깨 중앙을 회전축으로 가정, 반지름 비율로 클럽헤드 속도 추정
    """
    W = get_xyz_cols(df, wrist)         # (N,3) 좌표 원래 단위
    RA = get_xyz_cols(df, 'RAnkle')     # (N,3)
    LA = get_xyz_cols(df, 'LAnkle')     # (N,3)
    
    # 좌표 범위 로깅 (단위 확인용)
    coord_vals = W[~np.isnan(W)]
    coord_range_min = np.nanmin(np.abs(coord_vals)) if coord_vals.size > 0 else 0.0
    coord_range_max = np.nanmax(np.abs(coord_vals)) if coord_vals.size > 0 else 0.0
    print(f"[DEBUG] analyze_wrist_speed_3d: 좌표 범위 [{coord_range_min:.9f}, {coord_range_max:.6f}], scale_to_m={scale_to_m:.6f}")
    print(f"[DEBUG] scale_to_m={scale_to_m} → 예상 단위: {'mm' if scale_to_m==0.001 else 'cm' if scale_to_m==0.01 else 'm'}")
    
    wx = W[:, 0]
    stance_mid_x = (RA[:, 0] + LA[:, 0]) / 2.0
    
    # 3D 손목 속도 (m/s) - Z좌표 이상치 필터링 포함
    v_m_s = vectorized_speed_m_s_3d(W, fps, scale_to_m=scale_to_m, filter_z_outliers=True)
    print(f"[DEBUG] v_m_s 샘플 (처음 10프레임): {v_m_s[:10]}")
    
    # ⚠️ 추가 필터링: 프레임 간 속도가 비정상적으로 크면 제거
    # 정상: 0~100 km/h, 비정상: 100+ km/h 프레임
    v_ms, v_kmh, v_mph = _speed_conversions_m_s(v_m_s)
    
    # 비정상적인 속도 스파이크 검출 (IQR 기반)
    valid_speeds = v_kmh[np.isfinite(v_kmh)]
    if len(valid_speeds) > 4:
        q1 = np.percentile(valid_speeds, 25)
        q3 = np.percentile(valid_speeds, 75)
        iqr = q3 - q1
        speed_upper_bound = q3 + 3.0 * iqr  # 3 sigma 수준
        
        # 비정상 속도 프레임 마킹
        abnormal_mask = v_kmh > speed_upper_bound
        n_abnormal = np.sum(abnormal_mask)
        
        if n_abnormal > 0:
            abnormal_frames = np.where(abnormal_mask)[0]
            print(f"[WARN] 비정상 속도 프레임 감지: {n_abnormal}개 (limit={speed_upper_bound:.1f} km/h)")
            print(f"[WARN]   프레임: {list(abnormal_frames[:5])}{'...' if n_abnormal > 5 else ''}")
            print(f"[WARN]   속도: {v_kmh[abnormal_frames[:5]]} km/h")
            
            # 비정상 속도 프레임을 NaN으로 표시하여 보간
            v_m_s_for_interp = v_m_s.copy()
            v_m_s_for_interp[abnormal_mask] = np.nan
            
            # 보간 수행 (연속 비정상 프레임도 처리)
            valid_indices = np.where(~abnormal_mask)[0]
            if len(valid_indices) > 0:
                # 전체 배열을 Series로 변환하여 보간
                v_m_s_series = pd.Series(v_m_s_for_interp)
                v_m_s_interp = v_m_s_series.interpolate(method='linear', limit_direction='both').ffill().bfill().to_numpy()
                
                # 보간된 값만 업데이트
                v_m_s[abnormal_mask] = v_m_s_interp[abnormal_mask]
                
                # 모든 단위를 일관성 있게 재계산 (m/s 기반)
                v_ms, v_kmh, v_mph = _speed_conversions_m_s(v_m_s)
                
                interpolated_frames = abnormal_frames[v_m_s_interp[abnormal_frames] != v_m_s_for_interp[abnormal_frames]]
                for idx in interpolated_frames[:5]:
                    print(f"[DEBUG]   Frame {idx}: {v_kmh[idx]:.1f} km/h로 보간")
                if len(interpolated_frames) > 5:
                    print(f"[DEBUG]   ... 외 {len(interpolated_frames) - 5}개 프레임")
    
    print(f"[DEBUG] 필터링 후 v_m_s 샘플 (처음 10프레임): {v_m_s[:10]}")
    
    # 임팩트 프레임 탐지 (COM 기반 정확한 방식: RWrist X가 스탠스 중심을 +방향으로 교차)
    from impact_utils import detect_impact_by_crossing as detect_impact_com
    impact = detect_impact_com(df, prefer_2d=False, skip_ratio=0.0, smooth_window=3, hold_frames=0, margin=0.0)
    
    # ±2 프레임 내 피크 속도
    lo = max(0, impact - 2)
    hi = min(len(v_kmh) - 1, impact + 2)
    peak_local_idx = lo + int(np.nanargmax(v_kmh[lo:hi+1])) if hi >= lo else int(np.nanargmax(v_kmh))
    peak_wrist_kmh = float(v_kmh[peak_local_idx]) if not np.isnan(v_kmh[peak_local_idx]) else float(np.nanmax(v_kmh))
    peak_wrist_mph = float(v_mph[peak_local_idx]) if not np.isnan(v_mph[peak_local_idx]) else float(np.nanmax(v_mph))  # mph는 v_mph에서 직접 추출
    
    print(f"[DEBUG] Peak frame={peak_local_idx}, v_m_s={v_m_s[peak_local_idx]:.6f}, "
          f"v_km_h={peak_wrist_kmh:.2f}, v_mph={peak_wrist_mph:.2f}")
    
    # ---- 7번 아이언(37") 기준 각속도 기반 클럽 헤드 속도 추정 ----
    try:
        LSH = get_xyz_cols(df, 'LShoulder')  # (N,3)
        RSH = get_xyz_cols(df, 'RShoulder')  # (N,3)

        lsh_pt = LSH[peak_local_idx]
        rsh_pt = RSH[peak_local_idx]
        pivot_raw = 0.5 * (lsh_pt + rsh_pt)   # 회전축(원래 단위)

        wrist_raw = W[peak_local_idx]        # 원래 단위
        r_wrist_raw = float(np.linalg.norm(wrist_raw - pivot_raw))
        r_wrist_m = r_wrist_raw * float(scale_to_m)
        r_head_m = r_wrist_m + CLUB_LENGTH_7I_M

        scale = r_head_m / max(r_wrist_m, EPS_R)
        club_kmh = peak_wrist_kmh * scale
        club_mph = peak_wrist_mph * scale

        club_kmh_min = club_kmh_max = club_kmh
        club_mph_min = club_mph_max = club_mph
        effective_k = scale

    except Exception as e:
        print(f"[WARN] 3D 클럽 속도 각속도 기반 계산 실패, k-factor 폴백 사용: {e}")
        k = 1.35
        club_kmh = peak_wrist_kmh * k
        club_mph = peak_wrist_mph * k
        club_kmh_min = peak_wrist_kmh * 1.25
        club_kmh_max = peak_wrist_kmh * 1.55
        club_mph_min = peak_wrist_mph * 1.25
        club_mph_max = peak_wrist_mph * 1.55
        effective_k = k
        r_wrist_m = None
        r_head_m = None
        scale = None
    
    return {
        'impact_frame': int(impact),
        'peak_frame': int(peak_local_idx),
        'v_m_s': v_m_s,
        'v_km_h': v_kmh,
        'v_mph': v_mph,
        'wrist_peak_kmh': peak_wrist_kmh,
        'wrist_peak_mph': peak_wrist_mph,
        'club_kmh': club_kmh,
        'club_mph': club_mph,
        'club_kmh_range': (club_kmh_min, club_kmh_max),
        'club_mph_range': (club_mph_min, club_mph_max),
        'club_k_factor_effective': effective_k,
        'r_wrist_m': r_wrist_m,
        'r_head_m': r_head_m,
        'scale_r_head_over_r_wrist': scale,
    }

def analyze_wrist_speed_2d(df: pd.DataFrame, fps: int, wrist: str = "RWrist", m_per_px: Optional[float] = None):
    """
    입력: 2D CSV (px), 필수: {wrist}_x/{wrist}_y, RAnkle_x, LAnkle_x (있으면 사용)
    출력:
      - impact_frame, peak_frame
      - 시계열 속도 v_px_s
      - 피크 속도(손목) px/s
      - m_per_px 스케일이 있으면 m/s, km/h, mph도 함께 제공
      - 7번 아이언(37") 각속도 기반 클럽 헤드 속도 추정 (2D)
    """
    W = get_xy_cols_2d(df, wrist)        # (N,2) px
    RA = get_xy_cols_2d(df, 'RAnkle')     # (N,2) px (없으면 NaN)
    LA = get_xy_cols_2d(df, 'LAnkle')     # (N,2)
    wx = W[:, 0]
    stance_mid_x = (RA[:, 0] + LA[:, 0]) / 2.0
    
    # 2D 손목 속도 (px/s)
    v_px_s, unit = speed_2d(W, fps)
    
    # 임팩트 프레임 탐지 (COM 기반 정확한 방식: RWrist X가 스탠스 중심을 +방향으로 교차)
    from impact_utils import detect_impact_by_crossing as detect_impact_com
    impact = detect_impact_com(df, prefer_2d=True, skip_ratio=0.0, smooth_window=3, hold_frames=0, margin=0.0)
    
    # ±2 프레임 내 피크 속도
    lo = max(0, impact - 2)
    hi = min(len(v_px_s) - 1, impact + 2)
    peak_local_idx = lo + int(np.nanargmax(v_px_s[lo:hi+1])) if hi >= lo else int(np.nanargmax(v_px_s))
    peak_wrist_px_s = float(v_px_s[peak_local_idx]) if not np.isnan(v_px_s[peak_local_idx]) else float(np.nanmax(v_px_s))

    # 선택적: m/px 스케일이 주어지면 m/s로 환산하여 3D와 유사한 요약 제공
    if m_per_px is not None and m_per_px > 0:
        v_m_s = v_px_s * float(m_per_px)
        v_ms, v_kmh, v_mph = _speed_conversions_m_s(v_m_s)
        peak_wrist_kmh = float(v_kmh[peak_local_idx]) if not np.isnan(v_kmh[peak_local_idx]) else float(np.nanmax(v_kmh))
        peak_wrist_mph = peak_wrist_kmh / 1.609344  # km/h → mph 정확한 변환
        
        # ---- 7번 아이언(37") 기준 각속도 기반 클럽 헤드 속도 추정 ----
        try:
            LSH = get_xy_cols_2d(df, 'LShoulder')  # (N,2)
            RSH = get_xy_cols_2d(df, 'RShoulder')  # (N,2)
            lsh_pt = LSH[peak_local_idx]
            rsh_pt = RSH[peak_local_idx]
            pivot_px = 0.5 * (lsh_pt + rsh_pt)

            wrist_px = W[peak_local_idx]
            r_wrist_px = float(np.linalg.norm(wrist_px - pivot_px))
            r_wrist_m = r_wrist_px * float(m_per_px)
            r_head_m = r_wrist_m + CLUB_LENGTH_7I_M

            scale = r_head_m / max(r_wrist_m, EPS_R)
            club_kmh = peak_wrist_kmh * scale
            club_mph = peak_wrist_mph * scale

            club_kmh_min = club_kmh_max = club_kmh
            club_mph_min = club_mph_max = club_mph
            effective_k = scale

        except Exception as e:
            print(f"[WARN] 2D 클럽 속도 각속도 기반 계산 실패, k-factor 폴백 사용: {e}")
            k = 1.35
            club_kmh = peak_wrist_kmh * k
            club_mph = peak_wrist_mph * k
            club_kmh_min = peak_wrist_kmh * 1.25
            club_kmh_max = peak_wrist_kmh * 1.55
            club_mph_min = peak_wrist_mph * 1.25
            club_mph_max = peak_wrist_mph * 1.55
            effective_k = k
            r_wrist_m = None
            r_head_m = None
            scale = None
        
        return {
            'impact_frame': int(impact),
            'peak_frame': int(peak_local_idx),
            'v_px_s': v_px_s,
            'wrist_peak_px_s': peak_wrist_px_s,
            'v_m_s': v_m_s,
            'v_km_h': v_kmh,
            'v_mph': v_mph,
            'wrist_peak_kmh': peak_wrist_kmh,
            'wrist_peak_mph': peak_wrist_mph,
            'club_kmh': club_kmh,
            'club_mph': club_mph,
            'club_kmh_range': (club_kmh_min, club_kmh_max),
            'club_mph_range': (club_mph_min, club_mph_max),
            'unit': 'px/s',
            'calibrated_m_per_px': float(m_per_px),
            'club_k_factor_effective': effective_k,
            'r_wrist_m': r_wrist_m,
            'r_head_m': r_head_m,
            'scale_r_head_over_r_wrist': scale,
        }
    
    # 보정 불가 시 기존(px/s)만 반환
    return {
        'impact_frame': int(impact),
        'peak_frame': int(peak_local_idx),
        'v_px_s': v_px_s,
        'wrist_peak_px_s': peak_wrist_px_s,
        'unit': unit,
        'calibrated_m_per_px': None,
    }

def categorize_head_speed_mph(head_mph: float):
    """주어진 클럽 헤드 속도(mph)가 어떤 집단 평균에 가장 가까운지 멘트 구성"""
    refs = [
        ("Female Amateur", 78),
        ("Male Amateur", 93),
        ("LPGA Tour Pro", 94),
        ("PGA Tour Pro (avg male pro)", 114),
        ("Long Driver", 135),
        ("World Record", 157),
    ]
    # 가장 가까운 카테고리 선택
    best = min(refs, key=lambda kv: abs(head_mph - kv[1]))
    name, ref = best
    diff = head_mph - ref
    direction = "빠름" if diff >= 0 else "느림"
    return f"현재 추정 클럽 헤드 속도는 '{name}' 평균 {ref:.0f} mph와 가장 가깝습니다 (Δ{abs(diff):.1f} mph {direction})."

def load_cfg(p: Path):
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Use YAML for analyze config.")

def _coord_scale_to_m(cfg: dict) -> float:
    """분석 설정에서 좌표 단위 → 미터 환산 스케일을 결정합니다.
    
    우선순위 (높을수록 신뢰도 높음):
    1. intrinsics.json의 depth_scale (controller.py가 저장한 메타정보, 가장 신뢰도 높음)
       ⚠️ depth_scale은 RAW depth → MM 변환 계수입니다 (예: 0.001 = RAW × 0.001 = MM)
       controller.py가 CSV에 METER 단위로 저장하므로, scale_to_m = 1.0 적용
    2. analyze.yaml의 명시적 coord_unit 설정
    3. wide3 데이터 범위로부터 자동 감지
    4. 기본값: 1.0 (미터 단위, controller.py의 현재 표준)
    
    지원 단위: m, cm, mm (대소문자 무시)
    
    ⚠️ 중요: controller.py의 3D 처리는 이제 METER 단위로 저장합니다 (depth_scale과 무관).
    """
    # 1단계: intrinsics.json의 depth_scale (존재 여부 확인만, 값은 사용 안함)
    if 'intrinsics' in cfg and isinstance(cfg['intrinsics'], dict):
        meta = cfg['intrinsics'].get('meta', {})
        if isinstance(meta, dict):
            depth_scale = meta.get('depth_scale')
            if depth_scale is not None:
                try:
                    scale_val = float(depth_scale)
                    if scale_val > 0:
                        # depth_scale은 RAW→MM 계수이고, CSV는 METER 단위이므로
                        # scale_to_m = 1.0 (이미 미터 단위)
                        print(f"[INFO] ✅ LEVEL 1: intrinsics.depth_scale 감지 ({scale_val:.6f}), CSV는 METER 단위 → scale_to_m=1.0")
                        return 1.0
                except (TypeError, ValueError):
                    pass
    
    # 2단계: analyze.yaml의 coord_unit 명시 설정
    if 'coord_unit' in cfg:
        unit = (cfg.get("coord_unit") or "").strip().lower()
        if unit:
            if unit in ("m", "meter", "metre", "meters"):
                print(f"[INFO] ✅ LEVEL 2a: coord_unit='m' → scale_to_m=1.0")
                return 1.0
            if unit in ("cm", "centimeter", "centimetre", "centimeters"):
                print(f"[INFO] ✅ LEVEL 2b: coord_unit='cm' → scale_to_m=0.01")
                return 1e-2
            if unit in ("mm", "millimeter", "millimetre", "millimeters"):
                print(f"[INFO] ✅ LEVEL 2c: coord_unit='mm' → scale_to_m=0.001")
                return 1e-3
    
    # 3단계: wide3 데이터 범위로부터 자동 감지
    try:
        wide3 = cfg.get("wide3")
        if wide3 is not None and hasattr(wide3, 'columns'):
            # 3D 컬럼 찾기 (X3D/Y3D/Z3D 등 패턴)
            coord_cols = [c for c in wide3.columns if any(
                x.lower() in c.lower() for x in ('x3d', 'y3d', 'z3d', '__x', '__y', '__z', '_x', '_y', '_z')
            ) and (c.lower().endswith(('x3d', 'y3d', 'z3d', '__x', '__y', '__z', '_x', '_y', '_z')))]
            
            if coord_cols:
                print(f"[DEBUG] LEVEL 3: 감지된 3D 컬럼: {coord_cols[:3]}... (총 {len(coord_cols)}개)")
                all_vals = []
                for col in coord_cols:
                    try:
                        col_data = wide3[col].dropna()
                        if len(col_data) > 0:
                            all_vals.extend(col_data.abs().tolist())
                    except Exception:
                        pass
                
                if all_vals:
                    max_val = float(max(all_vals))
                    min_val = float(min([v for v in all_vals if v > 0]))
                    print(f"[DEBUG]   좌표 범위: [{min_val:.9f}, {max_val:.6f}]")
                    
                    # ⚠️ controller.py는 이제 METER 단위 저장하므로 범위 heuristic도 변경
                    # 미터 범위: 0.1 ~ 10
                    if min_val >= 0.001 and max_val <= 10:
                        print(f"[INFO] 🎯 LEVEL 3a: 미터 범위 좌표 → scale_to_m=1.0")
                        return 1.0
                    
                    # MM 범위 (구식 데이터): 50 이상
                    elif max_val >= 50:
                        print(f"[INFO] 🎯 LEVEL 3b: MM 범위 좌표 (max={max_val:.2f}) → scale_to_m=0.001")
                        return 1e-3
                    
                    # CM 범위: 0.1 ~ 10
                    elif min_val >= 0.1 and max_val <= 10:
                        print(f"[INFO] 🎯 LEVEL 3c: CM 범위 → scale_to_m=0.01")
                        return 0.01
                    
                    # 카메라 정규화 좌표: 0.0001 ~ 0.001
                    elif min_val >= 0.00001 and max_val < 0.001:
                        print(f"[INFO] 🎯 LEVEL 3d: 카메라 정규화 좌표 → scale_to_m=1.0")
                        return 1.0
                    
                    print(f"[WARN] 범위 판정 실패 [{min_val:.6f}, {max_val:.6f}], 기본값 미터 단위 적용")
            else:
                print(f"[DEBUG] 3D 컬럼 없음. 컬럼 샘플: {list(wide3.columns)[:5]}")
    except Exception as e:
        print(f"[DEBUG] 자동 감지 오류: {e}")
    
    # ⚠️ 최종 기본값: 1.0 (미터 단위)
    # controller.py가 항상 미터 단위로 저장하므로
    print(f"[INFO] 좌표 단위 미결정 → 기본값 미터 단위 적용 (scale_to_m=1.0)")
    return 1.0

# =========================================================
# Swing Speed 전용 계산 함수
# =========================================================
def get_swing_joints_2d(df: pd.DataFrame, wrist_r: str, wrist_l: str):
    """스윙에 관련된 관절들의 2D 좌표 확인"""
    swing_joints = [wrist_l, wrist_r]
    
    # 팔 관련 관절도 포함 (있다면)
    additional_joints = ["LShoulder", "RShoulder", "LElbow", "RElbow"]
    for joint in additional_joints:
        cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
        axes = cols_map.get(joint, {})
        if 'x' in axes and 'y' in axes:
            swing_joints.append(joint)
    
    print(f"🔗 Swing 관련 관절: {swing_joints}")
    return swing_joints

def build_swing_edges(kp_names):
    """스윙 관련 관절들만으로 연결선 생성"""
    E, have = [], set(kp_names)
    def add(a, b):
        if a in have and b in have: 
            E.append((a, b))
    
    # 팔 연결 (스윙의 핵심)
    add("LShoulder", "LElbow"); add("LElbow", "LWrist")
    add("RShoulder", "RElbow"); add("RElbow", "RWrist")
    
    # 어깨 연결
    add("LShoulder", "RShoulder")
    
    # 손목 연결 (그립 표시)
    add("LWrist", "RWrist")
    
    print(f"🔗 Swing용 연결선: {len(E)}개")
    return E

def compute_overlay_range(df: pd.DataFrame, kp_names):
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    xs, ys = [], []
    for name in kp_names:
        ax = cols_map.get(name, {})
        cx = ax.get('x'); cy = ax.get('y')
        if cx in df.columns: xs.extend(df[cx].dropna().tolist())
        if cy in df.columns: ys.extend(df[cy].dropna().tolist())
    if xs and ys:
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        small = all(abs(v) <= 2.0 for v in (x_min, x_max, y_min, y_max))
        print(f"📊 overlay 좌표 범위(swing): X({x_min:.4f}~{x_max:.4f}) Y({y_min:.4f}~{y_max:.4f}) smallRange={small}")
        return x_min, x_max, y_min, y_max, small
    print("⚠️ 좌표 데이터를 찾지 못했습니다. 픽셀 좌표로 간주합니다.")
    return None, None, None, None, False

# =========================================================
# Swing Speed 시각화 전용 오버레이
# =========================================================
def overlay_swing_video(
    img_dir: Path,
    df: pd.DataFrame,
    out_mp4: Path,
    fps: int,
    codec: str,
    wrist_r: str,
    wrist_l: str,
):
    """스윙 관련 관절들과 그립 포인트 시각화

    Note: 사용되지 않던 배열 인자(grip_points, R/L_points, 속도들)는 제거하고
    DataFrame 중심 API로 단순화했습니다.
    """
    images = sorted(glob.glob(str(img_dir / "*.png")), key=natural_key)
    if not images:
        images = sorted(glob.glob(str(img_dir / "*.jpg")), key=natural_key)
    if not images:
        images = sorted(glob.glob(str(img_dir / "*.jpeg")), key=natural_key)
    if not images:
        raise RuntimeError(f"No images (*.png|*.jpg|*.jpeg) in {img_dir}")

    first = cv2.imread(images[0])
    h, w = first.shape[:2]
    ensure_dir(out_mp4.parent)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
    
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {out_mp4}")

    # 스윙 관련 관절들만 시각화
    kp_names = get_swing_joints_2d(df, wrist_r, wrist_l)
    edges = build_swing_edges(kp_names)

    # 좌표 매핑 준비 (정규화 작은 범위면 화면으로 스케일)
    x_min, x_max, y_min, y_max, small = compute_overlay_range(df, kp_names)
    margin = 0.1
    def scale_xy(x, y):
        if np.isnan(x) or np.isnan(y):
            return np.nan, np.nan
        try:
            xf = float(x); yf = float(y)
        except Exception:
            return np.nan, np.nan
        if small and x_min is not None:
            dx = x_max - x_min if (x_max - x_min) != 0 else 1.0
            dy = y_max - y_min if (y_max - y_min) != 0 else 1.0
            x_norm = (xf - x_min) / dx
            y_norm = (yf - y_min) / dy
            sx = (margin + x_norm * (1 - 2*margin)) * w
            sy = (margin + y_norm * (1 - 2*margin)) * h
            return sx, sy
        return xf, yf
    
    # 그립 궤적 저장 (최근 50프레임)
    grip_trail = []

    # Prepare overlay-only dataframe: strong interpolation to remove (0,0) sentinel
    try:
        from .runner_utils import prepare_overlay_df
    except Exception:
        try:
            from runner_utils import prepare_overlay_df
        except Exception:
            prepare_overlay_df = None
    try:
        if prepare_overlay_df is not None and df is not None:
            df = prepare_overlay_df(df, prefer_2d=True, zero_threshold=0.0)
    except Exception:
        pass

    n_img = len(images)
    n_df = len(df)
    if n_img != n_df:
        print(f"⚠️ 프레임 개수 불일치(swing): images={n_img}, overlay_rows={n_df}. 이미지 길이에 맞춰 렌더링하고 CSV 부족분은 마지막 값을 재사용합니다.")

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        row_idx = i if i < n_df else (n_df - 1 if n_df > 0 else -1)
        row = df.iloc[row_idx] if row_idx >= 0 else None

        # --- 스윙 관절들 연결선 ---
        # import validity helper
        try:
            from .runner_utils import is_valid_overlay_point
        except Exception:
            try:
                from runner_utils import is_valid_overlay_point
            except Exception:
                is_valid_overlay_point = None

        for a, b in edges:
            ax, ay, _ = get_xyc_row(row, a)
            bx, by, _ = get_xyc_row(row, b)
            
            ax, ay = scale_xy(ax, ay)
            bx, by = scale_xy(bx, by)
            
            valid_ab = True
            if is_valid_overlay_point is not None:
                valid_ab = is_valid_overlay_point(ax, ay, w, h) and is_valid_overlay_point(bx, by, w, h)
            else:
                valid_ab = not (np.isnan(ax) or np.isnan(ay) or np.isnan(bx) or np.isnan(by)) and not ((ax == 0.0 and ay == 0.0) or (bx == 0.0 and by == 0.0))

            if valid_ab:
                # 손목 연결은 두껍게
                thickness = 4 if (a == wrist_l and b == wrist_r) else 2
                color = (0, 255, 0) if (a == wrist_l and b == wrist_r) else (0, 255, 255)
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), color, thickness)

        # --- 스윙 관절 점들 ---
        for name in kp_names:
            x, y, _ = get_xyc_row(row, name)
            x, y = scale_xy(x, y)
            valid_pt = True
            if is_valid_overlay_point is not None:
                valid_pt = is_valid_overlay_point(x, y, w, h)
            else:
                valid_pt = not (np.isnan(x) or np.isnan(y)) and not (x == 0.0 and y == 0.0)
            if valid_pt:
                # 손목은 크게, 다른 관절은 작게
                if name in [wrist_l, wrist_r]:
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)  # 빨간 큰 원
                    cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)  # 흰 테두리
                else:
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)  # 파란 작은 원

        # --- 그립 중심점 표시 (2D 좌표 기반) ---
        lx, ly, _ = get_xyc_row(row, wrist_l)
        rx, ry, _ = get_xyc_row(row, wrist_r)
        lx, ly = scale_xy(lx, ly)
        rx, ry = scale_xy(rx, ry)
        valid_grip = True
        if is_valid_overlay_point is not None:
            valid_grip = is_valid_overlay_point(lx, ly, w, h) and is_valid_overlay_point(rx, ry, w, h)
        else:
            valid_grip = not (np.isnan(lx) or np.isnan(ly) or np.isnan(rx) or np.isnan(ry)) and not ((lx == 0.0 and ly == 0.0) or (rx == 0.0 and ry == 0.0))

        if valid_grip:
            grip_x = (lx + rx) / 2.0
            grip_y = (ly + ry) / 2.0
            # 그립 중심점 (초록 다이아몬드)
            pts = np.array([
                [int(grip_x), int(grip_y-10)],
                [int(grip_x+10), int(grip_y)],
                [int(grip_x), int(grip_y+10)],
                [int(grip_x-10), int(grip_y)]
            ], np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 0))
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

            # 그립 궤적 추가
            grip_trail.append((int(grip_x), int(grip_y)))
            if len(grip_trail) > 50:  # 최근 50프레임만 유지
                grip_trail.pop(0)

            # 그립 궤적 그리기
            for j in range(1, len(grip_trail)):
                a = j / len(grip_trail)
                color_intensity = int(255 * a)
                cv2.line(frame, grip_trail[j-1], grip_trail[j], (color_intensity, 255, 0), 2)

        # HUD/텍스트/범례 제거: 영상엔 수치 표시 없음

        writer.write(frame)

    writer.release()

    # Ensure MP4 is browser-friendly: attempt H.264 transcode (libx264) + faststart, fallback to remux
    try:
        try:
            from .runner_utils import transcode_mp4_to_h264
        except Exception:
            try:
                from runner_utils import transcode_mp4_to_h264
            except Exception:
                transcode_mp4_to_h264 = lambda p, **kw: False
        try:
            ok = transcode_mp4_to_h264(str(out_mp4))
            if not ok:
                try:
                    from .runner_utils import ensure_mp4_faststart
                except Exception:
                    try:
                        from runner_utils import ensure_mp4_faststart
                    except Exception:
                        ensure_mp4_faststart = lambda p: False
                try:
                    ensure_mp4_faststart(str(out_mp4))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

# =========================================================
# run_from_context (프로그램적 실행 진입점)
# =========================================================
def run_from_context(ctx: dict):
    """Programmatic runner for swing_speed module (2D/3D 자동 분기).

    ctx(dict) 예상 키(선택적 포함):
      - dest_dir: 출력 루트 디렉토리 (기본 '.')
      - job_id | job: 작업 식별자 (파일 prefix)
      - wide2: 2D DataFrame (오버레이/2D 분석용)
      - wide3: 3D DataFrame (3D 분석용)
      - img_dir: 프레임 이미지 디렉토리
      - fps: 프레임 레이트 (기본 30)
      - codec: 비디오 코덱 (기본 'mp4v')
      - draw: {'smoothing': {...}} 2D 스무딩 옵션 (method, window, alpha 등)
      - landmarks: {'wrist_left': 'LWrist', 'wrist_right': 'RWrist'} 커스터마이즈 가능
      - coord_unit: 3D 좌표 단위(m|cm|mm) → 미터 환산
      - m_per_px_2d: 2D 보정 스케일 (m/px, 직접 지정)
      - calibration_2d: joint_pair 방식 캘리브레이션 dict
      - subject: {'shoulder_width_m': ..., 'height_m': ...} 자동 캘리브레이션 보조

    반환(dict):
      - metrics_csv: 메트릭 CSV 경로 또는 None
      - overlay_mp4: 스윙 오버레이 mp4 경로 또는 None
      - summary: 핵심 수치 요약(impact_frame, peak_frame, 손목/클럽 속도 등)
      - dimension: '2d' 또는 '3d'
      - errors: {'metrics': str?, 'overlay': str?} 실패 시
    """
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = str(ctx.get('job_id', ctx.get('job', 'job')))
        fps_ctx = ctx.get('fps')
        if fps_ctx is None:
            print(f"[WARN] swing_speed.run_from_context (secondary): fps not provided, using fallback 30")
            fps = 30
        else:
            fps = int(fps_ctx)
        wide3 = ctx.get('wide3')
        wide2 = ctx.get('wide2')
        if wide2 is None and wide3 is not None:
            # 2D 대체로 3D 재사용 가능 (overlay 최소 구현 위해)
            try:
                wide2 = wide3
            except Exception:
                wide2 = None
        img_dir = Path(ctx.get('img_dir', dest))
        codec = str(ctx.get('codec', 'mp4v'))
        lm = ctx.get('landmarks', {}) or {}
        wrist_l = lm.get('wrist_left', 'LWrist')
        wrist_r = lm.get('wrist_right', 'RWrist')
        ensure_dir(dest)

        out = {'metrics_csv': None, 'overlay_mp4': None, 'summary': {}, 'dimension': None, 'errors': {}}

        use_df = wide3 if wide3 is not None else wide2
        if use_df is not None:
            try:
                dim3 = is_dataframe_3d(use_df)
            except Exception:
                dim3 = False
            dimension = '3d' if dim3 else '2d'
            out['dimension'] = dimension
            try:
                if dimension == '3d':
                    # 3D 분석
                    # wide3과 intrinsics를 ctx에 추가하여 _coord_scale_to_m에서 활용 가능하게
                    ctx_for_scale = dict(ctx)
                    ctx_for_scale['wide3'] = use_df
                    
                    # intrinsics 정보 전달 (depth_scale 등 메타데이터 포함) - CRITICAL FOR CORRECT SCALING
                    print(f"\n[🔍 SCALE DETECTION START] 'intrinsics' in ctx = {'intrinsics' in ctx}")
                    if 'intrinsics' in ctx:
                        print(f"[DEBUG] intrinsics type = {type(ctx['intrinsics'])}")
                        if isinstance(ctx['intrinsics'], dict):
                            print(f"[DEBUG] intrinsics keys = {list(ctx['intrinsics'].keys())}")
                            meta = ctx['intrinsics'].get('meta', {})
                            if meta:
                                print(f"[DEBUG] intrinsics.meta keys = {list(meta.keys())}")
                                depth_scale = meta.get('depth_scale')
                                print(f"[DEBUG] ✅ intrinsics.meta.depth_scale = {depth_scale}")
                    
                    if 'intrinsics' in ctx and isinstance(ctx['intrinsics'], dict):
                        ctx_for_scale['intrinsics'] = ctx['intrinsics']
                        print(f"[DEBUG] ✅ intrinsics added to ctx_for_scale")
                    else:
                        print(f"[WARN] ❌ intrinsics NOT added (missing or not dict)")
                        print(f"[WARN]    Falling back to auto-detection from wide3 data range")
                    
                    scale_to_m = _coord_scale_to_m(ctx_for_scale)
                    print(f"[✅ SCALE DECISION] Final scale_to_m = {scale_to_m}")
                    print(f"[INFO] 이를 적용하면: 1 unit = {scale_to_m * 1000:.3f} mm")
                    if scale_to_m == 0.001:
                        print(f"[INFO] → 좌표가 MM 단위이고, 올바른 스케일입니다")
                    elif scale_to_m == 1.0:
                        print(f"[WARN] → 좌표가 M 단위로 간주됩니다 (검증 필요)")
                    print(f"[🔍 SCALE DETECTION END]\n")
                    
                    anal = analyze_wrist_speed_3d(use_df, fps=fps, wrist=wrist_r, scale_to_m=scale_to_m)
                    # 메트릭 CSV 구성 (프레임별 m/s, km/h, mph)
                    N = len(anal['v_m_s'])
                    metrics_df = pd.DataFrame({
                        'frame': range(N),
                        'wrist_speed_m_s': anal['v_m_s'],
                        'wrist_speed_km_h': anal['v_km_h'],
                        'wrist_speed_mph': anal['v_mph'],
                    })
                    effective_k = float(anal.get('club_k_factor_effective', 1.35))
                    summary = {
                        'impact_frame': int(anal['impact_frame']),
                        'peak_frame': int(anal['peak_frame']),
                        'wrist_peak_km_h': float(anal['wrist_peak_kmh']),
                        'wrist_peak_mph': float(anal['wrist_peak_mph']),
                        'club_k_factor': effective_k,
                        'club_speed_km_h': float(anal['club_kmh']),
                        'club_speed_mph': float(anal['club_mph']),
                        'club_speed_km_h_range': [float(anal['club_kmh_range'][0]), float(anal['club_kmh_range'][1])],
                        'club_speed_mph_range': [float(anal['club_mph_range'][0]), float(anal['club_mph_range'][1])],
                        'r_wrist_m': anal.get('r_wrist_m'),
                        'r_head_m': anal.get('r_head_m'),
                        'scale_r_head_over_r_wrist': anal.get('scale_r_head_over_r_wrist'),
                    }
                else:
                    # 2D 분석 + 선택적 보정
                    cfg_like = {
                        'm_per_px_2d': ctx.get('m_per_px_2d'),
                        'calibration_2d': ctx.get('calibration_2d'),
                        'subject': ctx.get('subject'),
                    }
                    m_per_px = _get_m_per_px_from_cfg(cfg_like, wide2) if wide2 is not None else None
                    anal = analyze_wrist_speed_2d(use_df, fps=fps, wrist=wrist_r, m_per_px=m_per_px)
                    if anal.get('calibrated_m_per_px'):
                        N = len(anal['v_m_s'])
                        metrics_df = pd.DataFrame({
                            'frame': range(N),
                            'wrist_speed_px_s': anal['v_px_s'],
                            'wrist_speed_m_s': anal['v_m_s'],
                            'wrist_speed_km_h': anal['v_km_h'],
                            'wrist_speed_mph': anal['v_mph'],
                        })
                        effective_k = float(anal.get('club_k_factor_effective', 1.35))
                        summary = {
                            'impact_frame': int(anal['impact_frame']),
                            'peak_frame': int(anal['peak_frame']),
                            'wrist_peak_km_h': float(anal['wrist_peak_kmh']),
                            'wrist_peak_mph': float(anal['wrist_peak_mph']),
                            'club_k_factor': effective_k,
                            'club_speed_km_h': float(anal['club_kmh']),
                            'club_speed_mph': float(anal['club_mph']),
                            'club_speed_km_h_range': [float(anal['club_kmh_range'][0]), float(anal['club_kmh_range'][1])],
                            'club_speed_mph_range': [float(anal['club_mph_range'][0]), float(anal['club_mph_range'][1])],
                            'calibrated_m_per_px': float(anal['calibrated_m_per_px']),
                            'r_wrist_m': anal.get('r_wrist_m'),
                            'r_head_m': anal.get('r_head_m'),
                            'scale_r_head_over_r_wrist': anal.get('scale_r_head_over_r_wrist'),
                        }
                    else:
                        N = len(anal['v_px_s'])
                        metrics_df = pd.DataFrame({
                            'frame': range(N),
                            'wrist_speed_px_s': anal['v_px_s'],
                        })
                        summary = {
                            'impact_frame': int(anal['impact_frame']),
                            'peak_frame': int(anal['peak_frame']),
                            'wrist_peak_px_s': float(anal['wrist_peak_px_s']),
                            'club_k_factor': 1.35,
                            'club_speed_km_h': None,
                            'club_speed_mph': None,
                            'club_speed_km_h_range': [None, None],
                            'club_speed_mph_range': [None, None],
                            'calibrated_m_per_px': None,
                        }
                # CSV 저장
                metrics_csv = dest / f"{job_id}_swing_speed_metrics.csv"
                ensure_dir(metrics_csv.parent)
                metrics_df.to_csv(metrics_csv, index=False)
                out['metrics_csv'] = str(metrics_csv)
                out['summary'] = summary
            except Exception as e:
                out['errors']['metrics'] = str(e)
        else:
            out['errors']['metrics'] = 'No DataFrame provided.'

        # ----------------------
        # Overlay 비디오 (2D 기반)
        # ----------------------
        overlay_path = dest / f"{job_id}_swing_speed_overlay.mp4"
        try:
            if wide2 is not None:
                draw_cfg = ctx.get('draw', {}) or {}
                smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
                if smooth_cfg.get('enabled', False):
                    method = smooth_cfg.get('method', 'ema')
                    window = int(smooth_cfg.get('window', 5))
                    alpha = float(smooth_cfg.get('alpha', 0.2))
                    gaussian_sigma = smooth_cfg.get('gaussian_sigma')
                    hampel_sigma = smooth_cfg.get('hampel_sigma', 3.0)
                    oneeuro_min_cutoff = smooth_cfg.get('oneeuro_min_cutoff', 1.0)
                    oneeuro_beta = smooth_cfg.get('oneeuro_beta', 0.007)
                    oneeuro_d_cutoff = smooth_cfg.get('oneeuro_d_cutoff', 1.0)
                    df_overlay_sm = smooth_df_2d(
                        wide2,
                        prefer_2d=True,
                        method=method,
                        window=window,
                        alpha=alpha,
                        fps=fps,
                        gaussian_sigma=gaussian_sigma,
                        hampel_sigma=hampel_sigma,
                        oneeuro_min_cutoff=oneeuro_min_cutoff,
                        oneeuro_beta=oneeuro_beta,
                        oneeuro_d_cutoff=oneeuro_d_cutoff,
                    )
                else:
                    df_overlay_sm = wide2
                overlay_swing_video(
                    img_dir=img_dir,
                    df=df_overlay_sm,
                    out_mp4=overlay_path,
                    fps=fps,
                    codec=codec,
                    wrist_r=wrist_r,
                    wrist_l=wrist_l,
                )
                out['overlay_mp4'] = str(overlay_path)
        except Exception as e:
            out['errors']['overlay'] = str(e)

        # Build full JSON matching main() and write to dest so CLI and run_from_context
        # produce identical output files.
        try:
            job_id_local = job_id if 'job_id' in locals() and job_id is not None else ctx.get('job_id', None)
            out_json = Path(dest) / "swing_speed_metric_result.json"
            # construct frames_obj similar to main()
            frames_obj = {}
            dimension = out.get('dimension') or out.get('summary', {}).get('dimension') or ('3d' if out.get('summary') and out.get('summary').get('wrist_peak_km_h') is not None else '2d')
            # prefer 'anal' if present (from computations above)
            if 'anal' in locals():
                if dimension == '3d':
                    N = len(anal.get('v_m_s', []))
                    for i in range(N):
                        vm = float(anal['v_m_s'][i]) if np.isfinite(anal['v_m_s'][i]) else None
                        vk = float(anal['v_km_h'][i]) if np.isfinite(anal['v_km_h'][i]) else None
                        vp = float(anal['v_mph'][i]) if np.isfinite(anal['v_mph'][i]) else None
                        frames_obj[str(i)] = {
                            'wrist_speed_m_s': vm,
                            'wrist_speed_km_h': vk,
                            'wrist_speed_mph': vp,
                        }
                    summary = out.get('summary', {})
                    out_obj = {
                        'job_id': job_id_local,
                        'dimension': '3d',
                        'overlay_mp4': out.get('overlay_mp4'),
                        'metrics': {
                            'swing_speed': {
                                'summary': summary,
                                'metrics_data': {
                                    'swing_speed_timeseries': frames_obj
                                }
                            }
                        }
                    }
                else:
                    # 2D: anal variable may be anal2d; handle calibrated vs px-only
                    anal2d = locals().get('anal2d') or locals().get('anal')
                    if anal2d is not None:
                        N = len(anal2d.get('v_px_s', []))
                        # if calibrated include m/s etc
                        if anal2d.get('calibrated_m_per_px'):
                            for i in range(N):
                                vpx = float(anal2d['v_px_s'][i]) if np.isfinite(anal2d['v_px_s'][i]) else None
                                vm = float(anal2d['v_m_s'][i]) if np.isfinite(anal2d['v_m_s'][i]) else None
                                vk = float(anal2d['v_km_h'][i]) if np.isfinite(anal2d['v_km_h'][i]) else None
                                vp = float(anal2d['v_mph'][i]) if np.isfinite(anal2d['v_mph'][i]) else None
                                frames_obj[str(i)] = {
                                    'wrist_speed_px_s': vpx,
                                    'wrist_speed_m_s': vm,
                                    'wrist_speed_km_h': vk,
                                    'wrist_speed_mph': vp,
                                }
                        else:
                            for i in range(N):
                                vpx = float(anal2d['v_px_s'][i]) if np.isfinite(anal2d['v_px_s'][i]) else None
                                frames_obj[str(i)] = {
                                    'wrist_speed_px_s': vpx,
                                    'wrist_speed_m_s': None,
                                    'wrist_speed_km_h': None,
                                    'wrist_speed_mph': None,
                                }
                        summary = out.get('summary', {})
                        out_obj = {
                            'job_id': job_id_local,
                            'dimension': '2d',
                            'overlay_mp4': out.get('overlay_mp4'),
                            'metrics': {
                                'swing_speed': {
                                    'summary': summary,
                                    'metrics_data': {
                                        'swing_speed_timeseries': frames_obj
                                    }
                                }
                            }
                        }
                    else:
                        out_obj = {'job_id': job_id_local, 'dimension': '2d', 'overlay_mp4': out.get('overlay_mp4'), 'metrics': {'swing_speed': {'summary': out.get('summary', {}), 'metrics_data': {'swing_speed_timeseries': {}}}}}
            else:
                out_obj = {'job_id': job_id_local, 'dimension': '2d', 'overlay_mp4': out.get('overlay_mp4'), 'metrics': {'swing_speed': {'summary': out.get('summary', {}), 'metrics_data': {'swing_speed_timeseries': {}}}}}

            try:
                out_json.write_text(__import__('json').dumps(out_obj, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass
            # return same dict
            return out_obj
        except Exception:
            # if building JSON fails, still return the original out dict
            return out

        return out
    except Exception as e:
        return {'error': str(e)}

# =========================================================
# 메인 함수
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Swing Speed 전용 분석기")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    args = ap.parse_args()
    
    cfg = load_cfg(Path(args.config))

    # CSV 분리: overlay(2D) vs metrics(3D) + 상호 폴백 허용
    overlay_csv = None
    metrics_csv = None
    if "overlay_csv_path" in cfg:
        overlay_csv = Path(cfg["overlay_csv_path"]); print(f"📊 Overlay(2D) CSV 사용(swing): {overlay_csv}")
    elif "csv_path" in cfg:
        overlay_csv = Path(cfg["csv_path"]); print(f"📊 Overlay(2D) CSV (fallback)(swing): {overlay_csv}")
    if "metrics_csv_path" in cfg:
        metrics_csv = Path(cfg["metrics_csv_path"]); print(f"📊 Metrics(3D) CSV 사용(swing): {metrics_csv}")
    elif "csv_path" in cfg:
        metrics_csv = Path(cfg["csv_path"]); print(f"📊 Metrics(3D) CSV (fallback)(swing): {metrics_csv}")
    img_dir = Path(cfg["img_dir"])
    fps = int(cfg.get("fps", 30))
    codec = str(cfg.get("codec", "mp4v"))
    
    # 손목 관절 이름
    lm_cfg = cfg.get("landmarks", {}) or {}
    wrist_l = lm_cfg.get("wrist_left", "LWrist")
    wrist_r = lm_cfg.get("wrist_right", "RWrist")
    
    # 출력 경로 (Swing 전용)
    out_csv = Path(cfg["metrics_csv"]).parent / "swing_speed_metrics.csv"
    out_mp4 = Path(cfg["overlay_mp4"]).parent / "swing_speed_analysis.mp4"

    # 1) CSV 로드 (서로 폴백)
    df_metrics = None
    df_overlay = None
    if metrics_csv is not None and metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)
        print(f"📋 Metrics CSV 로드(swing): {metrics_csv} ({len(df_metrics)} frames)")
    if overlay_csv is not None and overlay_csv.exists():
        df_overlay = pd.read_csv(overlay_csv)
        print(f"📋 Overlay CSV 로드(swing): {overlay_csv} ({len(df_overlay)} frames)")
    # 상호 폴백
    if df_metrics is None and df_overlay is not None:
        print("ℹ️ metrics CSV 없음 → overlay CSV를 metrics 용도로도 사용합니다.")
        df_metrics = df_overlay
    if df_overlay is None and df_metrics is not None:
        print("ℹ️ overlay CSV 없음 → metrics CSV를 overlay 용도로도 사용합니다.")
        df_overlay = df_metrics
    if df_metrics is None or df_overlay is None:
        raise RuntimeError("metrics/overlay CSV를 로드할 수 없습니다. analyze.yaml을 확인하세요.")

    # 2) 손목(RWrist) 기반 스윙 스피드 분석 (2D/3D 자동 분기)
    wrist_name = wrist_r  # 기본 Right wrist
    dim = "3d" if is_dataframe_3d(df_metrics) else "2d"
    if dim == "3d":
        scale_to_m = _coord_scale_to_m(cfg)
        print(f"🧭 좌표 단위 스케일: scale_to_m={scale_to_m:.6f} (m 기준)")
        anal3d = analyze_wrist_speed_3d(df_metrics, fps=fps, wrist=wrist_name, scale_to_m=scale_to_m)
    else:
        m_per_px = _get_m_per_px_from_cfg(cfg, df_overlay)
        if m_per_px is not None:
            print(f"🧭 2D 보정 사용: m_per_px={m_per_px:.6f} → px/s → m/s 변환")
        else:
            print("ℹ️ 2D 보정 스케일이 없어 px/s 단위로만 분석합니다. (config: m_per_px_2d 또는 calibration_2d 설정 가능)")
        anal2d = analyze_wrist_speed_2d(df_overlay, fps=fps, wrist=wrist_name, m_per_px=m_per_px)

    # 3) JSON 출력 준비 (xfactor와 동일 포맷)
    job_id = cfg.get("job_id")
    out_dir = Path(cfg.get("metrics_csv", metrics_csv)).parent
    ensure_dir(out_dir)
    out_json = out_dir / "swing_speed_metric_result.json"


    # 4) 비디오 오버레이 (2D 스무딩 적용 가능)
    draw_cfg = cfg.get('draw', {}) or {}
    smooth_cfg = (draw_cfg.get('smoothing') or {}) if isinstance(draw_cfg.get('smoothing'), dict) else {}
    if smooth_cfg.get('enabled', False):
        method = smooth_cfg.get('method', 'ema')
        window = int(smooth_cfg.get('window', 5))
        alpha = float(smooth_cfg.get('alpha', 0.2))
        gaussian_sigma = smooth_cfg.get('gaussian_sigma')
        hampel_sigma = smooth_cfg.get('hampel_sigma', 3.0)
        oneeuro_min_cutoff = smooth_cfg.get('oneeuro_min_cutoff', 1.0)
        oneeuro_beta = smooth_cfg.get('oneeuro_beta', 0.007)
        oneeuro_d_cutoff = smooth_cfg.get('oneeuro_d_cutoff', 1.0)
        df_overlay_sm = smooth_df_2d(
            df_overlay,
            prefer_2d=True,
            method=method,
            window=window,
            alpha=alpha,
            fps=fps,
            gaussian_sigma=gaussian_sigma,
            hampel_sigma=hampel_sigma,
            oneeuro_min_cutoff=oneeuro_min_cutoff,
            oneeuro_beta=oneeuro_beta,
            oneeuro_d_cutoff=oneeuro_d_cutoff,
        )
    else:
        df_overlay_sm = df_overlay

    overlay_swing_video(
        img_dir=img_dir,
        df=df_overlay_sm,
        out_mp4=out_mp4,
        fps=fps,
        codec=codec,
        wrist_r=wrist_r,
        wrist_l=wrist_l,
    )
    print(f"✅ Swing 분석 비디오 저장: {out_mp4}")

    # 5) 최종 출력 (JSON 일원화, xfactor 형식 준수)
    if dim == "3d":
        wrist_peak_mph = anal3d['wrist_peak_mph']
        wrist_peak_kmh = anal3d['wrist_peak_kmh']
        club_mph = anal3d['club_mph']
        club_kmh = anal3d['club_kmh']
        club_mph_min, club_mph_max = anal3d['club_mph_range']
        club_kmh_min, club_kmh_max = anal3d['club_kmh_range']
        effective_k = float(anal3d.get('club_k_factor_effective', 1.35))

        # 조언 멘트 (평균 Head Speed 표 기준)
        advice = categorize_head_speed_mph(club_mph)

        # 프레임별 시계열 구성
        frames_obj = {}
        N = len(anal3d['v_m_s'])
        for i in range(N):
            vm = float(anal3d['v_m_s'][i]) if np.isfinite(anal3d['v_m_s'][i]) else None
            vk = float(anal3d['v_km_h'][i]) if np.isfinite(anal3d['v_km_h'][i]) else None
            vp = float(anal3d['v_mph'][i]) if np.isfinite(anal3d['v_mph'][i]) else None
            frames_obj[str(i)] = {
                "wrist_speed_m_s": vm,
                "wrist_speed_km_h": vk,
                "wrist_speed_mph": vp,
            }

        out_obj = {
            "job_id": job_id,
            "dimension": "3d",
            "metrics": {
                "swing_speed": {
                    "summary": {
                        "impact_frame": int(anal3d['impact_frame']),
                        "peak_frame": int(anal3d['peak_frame']),
                        "wrist_peak_km_h": float(wrist_peak_kmh),
                        "wrist_peak_mph": float(wrist_peak_mph),
                        "club_k_factor": effective_k,
                        "club_speed_km_h": float(club_kmh),
                        "club_speed_mph": float(club_mph),
                        "club_speed_km_h_range": [float(club_kmh_min), float(club_kmh_max)],
                        "club_speed_mph_range": [float(club_mph_min), float(club_mph_max)],
                        "swing_speed_advice": [advice],
                        "unit": {
                            "timeseries_main": "m/s",
                            "timeseries_extras": ["km/h", "mph"]
                        }
                    },
                    "metrics_data": {
                        "swing_speed_timeseries": frames_obj
                    }
                }
            }
        }
    else:
        # 2D: 보정 여부에 따라 JSON 구성이 달라짐
        wrist_peak_px_s = anal2d['wrist_peak_px_s']
        N = len(anal2d['v_px_s'])
        frames_obj = {}
        if anal2d.get('calibrated_m_per_px'):
            # m/s 계열 포함
            for i in range(N):
                vpx = float(anal2d['v_px_s'][i]) if np.isfinite(anal2d['v_px_s'][i]) else None
                vm = float(anal2d['v_m_s'][i]) if np.isfinite(anal2d['v_m_s'][i]) else None
                vk = float(anal2d['v_km_h'][i]) if np.isfinite(anal2d['v_km_h'][i]) else None
                vp = float(anal2d['v_mph'][i]) if np.isfinite(anal2d['v_mph'][i]) else None
                frames_obj[str(i)] = {
                    "wrist_speed_px_s": vpx,
                    "wrist_speed_m_s": vm,
                    "wrist_speed_km_h": vk,
                    "wrist_speed_mph": vp,
                }
            wrist_peak_kmh = anal2d['wrist_peak_kmh']
            wrist_peak_mph = anal2d['wrist_peak_mph']
            club_kmh = anal2d['club_kmh']
            club_mph = anal2d['club_mph']
            club_kmh_min, club_kmh_max = anal2d['club_kmh_range']
            club_mph_min, club_mph_max = anal2d['club_mph_range']
            effective_k = float(anal2d.get('club_k_factor_effective', 1.35))
            advice = categorize_head_speed_mph(club_mph)
            out_obj = {
                "job_id": job_id,
                "dimension": "2d",
                "metrics": {
                    "swing_speed": {
                        "summary": {
                            "impact_frame": int(anal2d['impact_frame']),
                            "peak_frame": int(anal2d['peak_frame']),
                            "wrist_peak_km_h": float(wrist_peak_kmh),
                            "wrist_peak_mph": float(wrist_peak_mph),
                            "club_k_factor": effective_k,
                            "club_speed_km_h": float(club_kmh),
                            "club_speed_mph": float(club_mph),
                            "club_speed_km_h_range": [float(club_kmh_min), float(club_kmh_max)],
                            "club_speed_mph_range": [float(club_mph_min), float(club_mph_max)],
                            "swing_speed_advice": [advice],
                            "unit": {
                                "timeseries_main": "m/s",
                                "timeseries_extras": ["km/h", "mph", "px/s"],
                                "calibrated_m_per_px": float(anal2d['calibrated_m_per_px'])
                            }
                        },
                        "metrics_data": {
                            "swing_speed_timeseries": frames_obj
                        }
                    }
                }
            }
        else:
            # px/s만 제공
            for i in range(N):
                vpx = float(anal2d['v_px_s'][i]) if np.isfinite(anal2d['v_px_s'][i]) else None
                frames_obj[str(i)] = {
                    "wrist_speed_px_s": vpx,
                    "wrist_speed_m_s": None,
                    "wrist_speed_km_h": None,
                    "wrist_speed_mph": None,
                }
            out_obj = {
                "job_id": job_id,
                "dimension": "2d",
                "metrics": {
                    "swing_speed": {
                        "summary": {
                            "impact_frame": int(anal2d['impact_frame']),
                            "peak_frame": int(anal2d['peak_frame']),
                            "wrist_peak_km_h": None,
                            "wrist_peak_mph": None,
                            "club_k_factor": 1.35,
                            "club_speed_km_h": None,
                            "club_speed_mph": None,
                            "club_speed_km_h_range": [None, None],
                            "club_speed_mph_range": [None, None],
                            "swing_speed_advice": [],
                            "unit": {
                                "timeseries_main": "px/s",
                                "timeseries_extras": []
                            }
                        },
                        "metrics_data": {
                            "swing_speed_timeseries": frames_obj
                        }
                    }
                }
            }

    out_json.write_text(__import__('json').dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Swing Speed JSON 저장: {out_json}")

    # 콘솔 요약
    print("\n결과")
    if dim == "3d":
        effective_k = float(anal3d.get('club_k_factor_effective', 1.35))
        print(f"실제 swing speed (손목) : {wrist_peak_kmh:.1f} km/h ({wrist_peak_mph:.1f} mph)")
        print(f"추정 club speed (클럽) : {club_kmh:.1f} km/h ({club_mph:.1f} mph)  [k={effective_k:.2f}, 범위 {club_kmh_min:.1f}~{club_kmh_max:.1f} km/h]")
        print(f"📝 조언: {advice}")
    else:
        if anal2d.get('calibrated_m_per_px'):
            effective_k = float(anal2d.get('club_k_factor_effective', 1.35))
            print(f"실제 swing speed (손목) : {wrist_peak_kmh:.1f} km/h ({wrist_peak_mph:.1f} mph) [2D 보정]  (m_per_px={anal2d['calibrated_m_per_px']:.6f})")
            print(f"추정 club speed (클럽) : {club_kmh:.1f} km/h ({club_mph:.1f} mph)  [k={effective_k:.2f}, 범위 {club_kmh_min:.1f}~{club_kmh_max:.1f} km/h]")
            print(f"📝 조언: {advice}")
        else:
            print(f"실제 swing speed (손목) : {wrist_peak_px_s:.1f} px/s (2D, 보정 없음)")

if __name__ == "__main__":
    main()
