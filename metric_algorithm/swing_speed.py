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
from pathlib import Path as _P
sys.path.append(str(_P(__file__).parent))

from utils_io import natural_key, ensure_dir
from impact_utils import detect_impact_by_crossing, compute_stance_mid_and_width as compute_stance_utils
from runner_utils import (
    parse_joint_axis_map_from_columns,
    is_dataframe_3d,
    get_xyz_cols,
    get_xy_cols_2d,
    get_xyc_row,
    scale_xy_for_overlay,
    compute_overlay_range as _compute_overlay_range_base,
    write_df_csv,
    images_to_mp4,
    upload_overlay_to_s3,
    normalize_result,
)

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
    half = window // 2
    xs = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    s = _interpolate_series(pd.Series(arr))
    y = np.convolve(s.to_numpy(), kernel, mode='same')
    return y


def _hampel(arr: np.ndarray, window: int, n_sigmas: float, alpha: float) -> np.ndarray:
    if window <= 1:
        return arr
    s = pd.Series(arr)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
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
            y_series = pd.Series(y, index=s.index)
            y_series[s.isna()] = np.nan
            out[col] = y_series
    print(f"🎛️ 2D 스무딩 적용: method={method}, window={window}, alpha={alpha}")
    return out

# =========================================================
# 3D 속도 및 이상치 필터링
# =========================================================
def _filter_depth_outliers(z_coords: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Z 좌표의 이상치를 필터링합니다 (depth 추적 오류 제거).
    
    세 단계 필터링:
    1. 절대값 범위: 시연 환경 기준 0.5m~2.5m 벗어나면 제거
    2. 급격한 점프: 프레임간 1.5m 이상 변화시 제거
    3. IQR 기반: ±2.5 IQR 벗어나는 값 제거
    """
    z = z_coords.astype(float).copy()
    valid_mask = np.isfinite(z)
    if not np.any(valid_mask):
        return z

    # 1단계: 절대값 범위 (시연 환경 물리적 제약)
    DEPTH_MIN = 0.5  # 0.5m 이하는 너무 가까움
    DEPTH_MAX = 3.0  # 극단적인 이상치만 제거 (2.0m은 너무 공격적)
    abs_outlier_mask = (z < DEPTH_MIN) | (z > DEPTH_MAX)
    z[abs_outlier_mask] = np.nan
    n_abs_outliers = np.sum(abs_outlier_mask)
    
    # 2단계: 급격한 점프 감지 (프레임간 연속성)
    dz = np.abs(np.diff(z, prepend=z[0]))
    JUMP_THRESHOLD = 1.5  # 60fps 기준 프레임간 1.5m 이상 변화는 비정상
    jump_mask = (dz > JUMP_THRESHOLD) & np.isfinite(z)
    z[jump_mask] = np.nan
    n_jump_outliers = np.sum(jump_mask)
    
    # 3단계: IQR 기반 (남은 유효값 중 통계적 이상치)
    valid_mask = np.isfinite(z)
    if not np.any(valid_mask):
        return z
    
    valid_vals = z[valid_mask]
    if len(valid_vals) < 4:  # IQR 계산 불가능
        return z
        
    q1 = np.percentile(valid_vals, 25)
    q3 = np.percentile(valid_vals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 2.5 * iqr
    upper_bound = q3 + 2.5 * iqr
    outlier_mask = (z < lower_bound) | (z > upper_bound)
    n_iqr_outliers = np.sum(outlier_mask & np.isfinite(z))

    if verbose and (n_abs_outliers > 0 or n_jump_outliers > 0 or n_iqr_outliers > 0):
        print(f"[DEBUG] 절대값 범위 이상치: {n_abs_outliers}개 (허용: {DEPTH_MIN}-{DEPTH_MAX}m)")
        print(f"[DEBUG] 급격한 점프 이상치: {n_jump_outliers}개 (임계값: {JUMP_THRESHOLD}m)")
        print(f"[DEBUG] IQR 통계 이상치: {n_iqr_outliers}개 (범위: [{lower_bound:.2f}, {upper_bound:.2f}]m)")
        if n_abs_outliers > 0:
            abs_indices = np.where(abs_outlier_mask)[0]
            print(f"[DEBUG] 범위 초과 프레임: {list(abs_indices[:10])}")
        if n_jump_outliers > 0:
            jump_indices = np.where(jump_mask)[0]
            print(f"[DEBUG] 점프 이상 프레임: {list(jump_indices[:10])}")

    z[outlier_mask] = np.nan
    # CRITICAL: Do NOT interpolate here - already done in controller.py and vectorized_speed_m_s_3d
    # Multiple interpolations spread outliers to neighboring frames
    return z


def vectorized_speed_m_s_3d(points_xyz: np.ndarray, fps: int, scale_to_m: float = 1.0,
                            filter_z_outliers: bool = False) -> np.ndarray:
    """
    벡터화된 손목 3D 속도(m/s) 계산
    v = Δs * fps (Δs는 좌표 단위 거리, scale_to_m로 m로 환산)
    
    NOTE: filter_z_outliers는 deprecated - controller.py에서 skeleton3d.csv 생성 시 이미 필터링됨
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        return np.full((len(points_xyz),), np.nan, dtype=float)
    X = points_xyz.astype(float).copy()

    # DEPRECATED: Z-axis filtering now done in controller.py before CSV creation
    # All metrics share the same filtered skeleton3d.csv data (atomic operation)
    if filter_z_outliers:
        print(f"[WARN] filter_z_outliers=True is deprecated - filtering already applied in skeleton3d.csv")

    # CRITICAL: Single interpolation pass (skeleton3d.csv is already filtered but may have NaN gaps)
    for c in range(3):
        s = pd.Series(X[:, c])
        s = s.interpolate(limit_direction='both').ffill().bfill()
        X[:, c] = s.to_numpy()

    dx = np.diff(X[:, 0], prepend=X[0, 0])
    dy = np.diff(X[:, 1], prepend=X[0, 1])
    dz = np.diff(X[:, 2], prepend=X[0, 2])
    ds = np.sqrt(dx**2 + dy**2 + dz**2)

    ds_m = ds * float(scale_to_m)
    fps_float = float(fps if fps and fps > 0 else 30)
    v_m_s = ds_m * fps_float
    if len(v_m_s) > 0:
        v_m_s[0] = 0.0
    return v_m_s


def _speed_conversions_m_s(v_m_s: np.ndarray):
    """m/s 배열을 km/h, mph로 동시 변환"""
    KM_H_PER_M_S = 3.6
    MPH_PER_M_S = 3.6 / 1.609344
    v_kmh = v_m_s * KM_H_PER_M_S
    v_mph = v_m_s * MPH_PER_M_S
    return v_m_s, v_kmh, v_mph

# =========================================================
# 2D/3D 공통 헬퍼
# =========================================================
def is_dataframe_3d(df: pd.DataFrame) -> bool:
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=False)
    for axes in cols_map.values():
        if 'z' in axes:
            return True
    return False


def get_xy_cols_2d(df: pd.DataFrame, name: str) -> np.ndarray:
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    if name in cols_map and all(a in cols_map[name] for a in ('x', 'y')):
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

# =========================================================
# 캘리브레이션 및 스케일링
# =========================================================
def _pair_distance_px_series_2d(df: pd.DataFrame, joint_a: str, joint_b: str) -> np.ndarray:
    """2D에서 두 관절 사이의 프레임별 거리(px) 시계열."""
    A = get_xy_cols_2d(df, joint_a)
    B = get_xy_cols_2d(df, joint_b)
    for arr in (A, B):
        for c in range(arr.shape[1]):
            s = pd.Series(arr[:, c])
            s = s.interpolate(limit_direction='both').ffill().bfill()
            arr[:, c] = s.to_numpy()
    d = np.sqrt((A[:, 0] - B[:, 0])**2 + (A[:, 1] - B[:, 1])**2)
    return d


def _sanity_check_m_per_px(df: pd.DataFrame, m_per_px: Optional[float]) -> Optional[float]:
    """
    2D 캘리브레이션 후, 어깨폭 기준으로 m_per_px가 말이 되는지 재검증.
    - 성인 어깨폭 정상 범위: 0.30 ~ 0.60 m
    - 범위 밖이면 L/RShoulder median을 기준으로 어깨폭 0.45 m가 되도록 m_per_px 재조정.
    """
    if m_per_px is None or m_per_px <= 0:
        return m_per_px

    try:
        d = _pair_distance_px_series_2d(df, "LShoulder", "RShoulder")
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size == 0:
            return m_per_px

        px_med = float(np.median(valid))
        shoulder_m = px_med * float(m_per_px)

        MIN_SHOULDER = 0.30
        MAX_SHOULDER = 0.60
        TARGET_SHOULDER = 0.45

        if shoulder_m < MIN_SHOULDER or shoulder_m > MAX_SHOULDER:
            new_m_per_px = TARGET_SHOULDER / px_med
            print(
                f"⚠️ 2D 캘리브레이션 sanity check: "
                f"현재 어깨폭≈{shoulder_m:.3f}m (px_med={px_med:.2f}, m_per_px={m_per_px:.6f}) → "
                f"비정상 → m_per_px={new_m_per_px:.6f}로 재조정"
            )
            return new_m_per_px
    except Exception as e:
        print(f"[WARN] _sanity_check_m_per_px 실패: {e}")

    return m_per_px


def _autocalibrate_m_per_px(df: pd.DataFrame, cfg: dict) -> Optional[float]:
    """
    피사체 신체 비율 기반의 자동 캘리브레이션.
    - 후보 관절쌍 중 "어깨 우선"으로 선택:
      shoulder > hip > ankle
    - 실제 길이:
        1) subject.shoulder_width_m
        2) subject.height_m * 0.259
        3) 기본값 0.40 m
    """
    candidates = [
        ("LShoulder", "RShoulder", "shoulder"),
        ("LHip", "RHip", "hip"),
        ("LAnkle", "RAnkle", "ankle"),
    ]
    stats = []
    for a, b, tag in candidates:
        d = _pair_distance_px_series_2d(df, a, b)
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size == 0:
            continue
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med))) if valid.size > 0 else 0.0
        cv = (mad / med) if med > 1e-6 else 1e9
        stats.append((a, b, tag, med, cv))

    if not stats:
        return None

    preferred_rank = {"shoulder": 0, "hip": 1, "ankle": 2}
    stats.sort(key=lambda x: (preferred_rank.get(x[2], 99), x[4], -x[3]))
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
                real_len_m = 0.259 * h
        except Exception:
            pass
    if real_len_m is None:
        real_len_m = 0.40

    m_per_px = real_len_m / px_med if px_med > 0 else None
    if m_per_px is not None:
        print(f"🧭 2D 자동 보정: pair={a}-{b} median={px_med:.2f}px, real≈{real_len_m:.3f}m → m_per_px={m_per_px:.6f} (cv={cv:.3f})")
    return m_per_px


def _get_m_per_px_from_cfg(cfg: dict, df_overlay: pd.DataFrame) -> Optional[float]:
    """
    analyze.yaml + 자동 캘리브레이션에서 2D 보정 스케일(m/px) 결정.
    우선순위:
      1. calibration_2d.joint_pair
      2. 자동 캘리브레이션(auto)
      3. m_per_px_2d 직접 지정
    마지막에 sanity check로 어깨폭 범위를 강제.
    """
    calib = cfg.get("calibration_2d") or {}

    # 1) joint_pair 수동 캘리브레이션
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
                    px_med = float(np.median(d_px_valid))
                    m_per_px = real_len_m / px_med
                    print(f"🧭 [우선순위 1] 수동 관절 쌍 캘리브레이션: {ja}-{jb} median={px_med:.2f}px, real={real_len_m:.3f}m → m_per_px={m_per_px:.6f}")
                    return _sanity_check_m_per_px(df_overlay, m_per_px)
                else:
                    print("⚠️ 캘리브레이션용 관절 쌍 거리(px)를 계산할 수 없습니다.")

    # 2) 자동 캘리브레이션
    auto_flag = True if calib.get("method", "").lower() in ("", "auto") else False
    if auto_flag:
        mpp_auto = _autocalibrate_m_per_px(df_overlay, cfg)
        if mpp_auto is not None:
            return _sanity_check_m_per_px(df_overlay, mpp_auto)

    # 3) 직접 지정
    mpp = cfg.get("m_per_px_2d")
    if mpp is not None:
        try:
            val = float(mpp)
            if val > 0:
                print(f"🧭 [우선순위 3] 2D 보정 스케일 직접 지정: m_per_px={val:.6f}")
                return _sanity_check_m_per_px(df_overlay, val)
        except Exception:
            pass

    return None


def _coord_scale_to_m(cfg: dict) -> float:
    """
    3D 좌표 단위 → m로 환산 스케일.
    기본적으로 controller가 m단위로 저장한다고 가정 → 기본값 1.0
    """
    if 'intrinsics' in cfg and isinstance(cfg['intrinsics'], dict):
        meta = cfg['intrinsics'].get('meta', {})
        if isinstance(meta, dict):
            depth_scale = meta.get('depth_scale')
            if depth_scale is not None:
                try:
                    scale_val = float(depth_scale)
                    if scale_val > 0:
                        print(f"[INFO] ✅ LEVEL 1: intrinsics.depth_scale 감지 ({scale_val:.6f}), CSV는 METER 단위 → scale_to_m=1.0")
                        return 1.0
                except (TypeError, ValueError):
                    pass

    unit = (cfg.get("coord_unit") or "").strip().lower()
    if unit:
        if unit in ("m", "meter", "metre", "meters"):
            print(f"[INFO] ✅ coord_unit='m' → scale_to_m=1.0")
            return 1.0
        if unit in ("cm", "centimeter", "centimetre", "centimeters"):
            print(f"[INFO] ✅ coord_unit='cm' → scale_to_m=0.01")
            return 1e-2
        if unit in ("mm", "millimeter", "millimetre", "millimeters"):
            print(f"[INFO] ✅ coord_unit='mm' → scale_to_m=0.001")
            return 1e-3

    try:
        wide3 = cfg.get("wide3")
        if wide3 is not None and hasattr(wide3, 'columns'):
            coord_cols = [c for c in wide3.columns if any(
                x.lower() in c.lower() for x in ('x3d', 'y3d', 'z3d')
            )]
            if coord_cols:
                all_vals = []
                for col in coord_cols:
                    col_data = wide3[col].dropna()
                    if len(col_data) > 0:
                        all_vals.extend(col_data.abs().tolist())
                if all_vals:
                    max_val = float(max(all_vals))
                    min_pos = [v for v in all_vals if v > 0]
                    min_val = float(min(min_pos)) if min_pos else 0.0
                    print(f"[DEBUG] 좌표 범위: [{min_val:.9f}, {max_val:.6f}]")

                    if min_val >= 0.001 and max_val <= 10:
                        print(f"[INFO] 🎯 미터 범위 좌표 → scale_to_m=1.0")
                        return 1.0
                    elif max_val >= 50:
                        print(f"[INFO] 🎯 MM 범위 좌표 → scale_to_m=0.001")
                        return 1e-3
                    elif min_val >= 0.1 and max_val <= 10:
                        print(f"[INFO] 🎯 CM 범위 → scale_to_m=0.01")
                        return 0.01
                    elif min_val >= 0.00001 and max_val < 0.001:
                        print(f"[INFO] 🎯 카메라 정규화 좌표 → scale_to_m=1.0")
                        return 1.0
    except Exception as e:
        print(f"[DEBUG] 자동 감지 오류: {e}")

    print(f"[INFO] 좌표 단위 미결정 → 기본값 미터 단위 적용 (scale_to_m=1.0)")
    return 1.0

# =========================================================
# 손목→클럽 속도 매핑 (7번 아이언 기준)
# =========================================================
def _estimate_shoulder_width_m(df: pd.DataFrame, m_per_px: Optional[float]) -> Optional[float]:
    """2D에서 어깨폭(m) 추정."""
    if m_per_px is None or m_per_px <= 0:
        return None
    d = _pair_distance_px_series_2d(df, "LShoulder", "RShoulder")
    valid = d[np.isfinite(d) & (d > 0)]
    if valid.size == 0:
        return None
    px_med = float(np.median(valid))
    return px_med * float(m_per_px)


def _estimate_shoulder_width_m_3d(df: pd.DataFrame) -> Optional[float]:
    """3D에서 어깨폭(m) 추정."""
    try:
        LS = get_xyz_cols(df, "LShoulder")
        RS = get_xyz_cols(df, "RShoulder")
        d = np.sqrt(np.sum((LS - RS) ** 2, axis=1))
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))
    except Exception:
        return None


def estimate_club_speed_7i_from_wrist(
    wrist_peak_kmh: float,
    wrist_peak_mph: float,
    shoulder_width_m: Optional[float] = None,
):
    """
    항상 7번 아이언(37 inch ≈ 0.94m) 기준으로
    손목 속도 → 클럽 속도 변환.

    기본 가정:
      - 어깨→손목: 0.65 m
      - 7i 길이:   0.94 m
      => k_base ≈ (0.65 + 0.94) / 0.65 ≈ 2.45

    + 어깨폭에 따라 ±20% 이내에서만 조정
    + 최종 k는 1.8 ~ 3.2 범위로 클램프
    """
    ARM_LEN_REF = 0.65
    CLUB_LEN_7I = 0.94
    K_BASE = (ARM_LEN_REF + CLUB_LEN_7I) / ARM_LEN_REF  # ≈ 2.45

    k = K_BASE
    if shoulder_width_m is not None:
        ratio = shoulder_width_m / 0.45  # 기준 어깨폭 0.45m
        ratio = max(0.8, min(1.2, ratio))
        k *= ratio

    k = max(1.8, min(3.2, k))

    club_kmh = wrist_peak_kmh * k
    club_mph = wrist_peak_mph * k

    k_min = max(1.8, k - 0.3)
    k_max = min(3.2, k + 0.3)
    club_kmh_min, club_kmh_max = wrist_peak_kmh * k_min, wrist_peak_kmh * k_max
    club_mph_min, club_mph_max = wrist_peak_mph * k_min, wrist_peak_mph * k_max

    return {
        "club_k_factor": k,
        "club_speed_km_h": club_kmh,
        "club_speed_mph": club_mph,
        "club_speed_km_h_range": (club_kmh_min, club_kmh_max),
        "club_speed_mph_range": (club_mph_min, club_mph_max),
    }

# =========================================================
# 3D 분석
# =========================================================
def analyze_wrist_speed_3d(df: pd.DataFrame, fps: int, wrist: str = "RWrist", scale_to_m: float = 1.0):
    """
    3D 손목 속도 분석 + 7번 아이언 기준 클럽 속도 추정.
    
    NOTE: Z-axis filtering is now done in controller.py when creating skeleton3d.csv
    All metrics share the same filtered data for consistency (atomic operation)
    """
    W = get_xyz_cols(df, wrist)
    RA = get_xyz_cols(df, 'RAnkle')
    LA = get_xyz_cols(df, 'LAnkle')

    coord_range_min = np.nanmin(np.abs(W[~np.isnan(W)]))
    coord_range_max = np.nanmax(np.abs(W[~np.isnan(W)]))
    print(f"[DEBUG] analyze_wrist_speed_3d: 좌표 범위 [{coord_range_min:.9f}, {coord_range_max:.6f}], scale_to_m={scale_to_m:.6f}")

    # filter_z_outliers=False: filtering already done in skeleton3d.csv creation
    v_m_s = vectorized_speed_m_s_3d(W, fps, scale_to_m=scale_to_m, filter_z_outliers=False)
    print(f"[DEBUG] v_m_s 샘플 (처음 10프레임): {v_m_s[:10]}")

    v_ms, v_kmh, v_mph = _speed_conversions_m_s(v_m_s)

    valid_speeds = v_kmh[np.isfinite(v_kmh)]
    if len(valid_speeds) > 4:
        q1 = np.percentile(valid_speeds, 25)
        q3 = np.percentile(valid_speeds, 75)
        iqr = q3 - q1
        speed_upper_bound = q3 + 3.0 * iqr
        abnormal_mask = v_kmh > speed_upper_bound
        n_abnormal = np.sum(abnormal_mask)
        if n_abnormal > 0:
            abnormal_frames = np.where(abnormal_mask)[0]
            print(f"[WARN] 비정상 속도 프레임 감지: {n_abnormal}개 (limit={speed_upper_bound:.1f} km/h)")
            print(f"[WARN]   프레임: {list(abnormal_frames[:5])}{'...' if n_abnormal > 5 else ''}")
            v_m_s_for_interp = v_m_s.copy()
            v_m_s_for_interp[abnormal_mask] = np.nan
            v_m_s_series = pd.Series(v_m_s_for_interp)
            v_m_s_interp = v_m_s_series.interpolate(method='linear', limit_direction='both').ffill().bfill().to_numpy()
            v_m_s[abnormal_mask] = v_m_s_interp[abnormal_mask]
            v_ms, v_kmh, v_mph = _speed_conversions_m_s(v_m_s)

    print(f"[DEBUG] 필터링 후 v_m_s 샘플 (처음 10프레임): {v_m_s[:10]}")

    impact = detect_impact_by_crossing(df, prefer_2d=False, skip_ratio=0.0, smooth_window=3, hold_frames=0, margin=0.0)

    lo = max(0, impact - 2)
    hi = min(len(v_kmh) - 1, impact + 2)
    peak_local_idx = lo + int(np.nanargmax(v_kmh[lo:hi+1])) if hi >= lo else int(np.nanargmax(v_kmh))
    peak_wrist_kmh = float(v_kmh[peak_local_idx]) if not np.isnan(v_kmh[peak_local_idx]) else float(np.nanmax(v_kmh))
    peak_wrist_mph = float(v_mph[peak_local_idx]) if not np.isnan(v_mph[peak_local_idx]) else float(np.nanmax(v_mph))

    print(f"[DEBUG] Peak frame={peak_local_idx}, v_m_s={v_m_s[peak_local_idx]:.6f}, "
          f"v_km_h={peak_wrist_kmh:.2f}, v_mph={peak_wrist_mph:.2f}")

    shoulder_width_m = _estimate_shoulder_width_m_3d(df)
    club_info = estimate_club_speed_7i_from_wrist(
        wrist_peak_kmh=peak_wrist_kmh,
        wrist_peak_mph=peak_wrist_mph,
        shoulder_width_m=shoulder_width_m,
    )

    return {
        'impact_frame': int(impact),
        'peak_frame': int(peak_local_idx),
        'v_m_s': v_m_s,
        'v_km_h': v_kmh,
        'v_mph': v_mph,
        'wrist_peak_kmh': peak_wrist_kmh,
        'wrist_peak_mph': peak_wrist_mph,
        'club_k_factor': float(club_info["club_k_factor"]),
        'club_kmh': float(club_info["club_speed_km_h"]),
        'club_mph': float(club_info["club_speed_mph"]),
        'club_kmh_range': tuple(map(float, club_info["club_speed_km_h_range"])),
        'club_mph_range': tuple(map(float, club_info["club_speed_mph_range"])),
    }

# =========================================================
# 2D 분석
# =========================================================
def analyze_wrist_speed_2d(df: pd.DataFrame, fps: int, wrist: str = "RWrist", m_per_px: Optional[float] = None):
    """
    2D 손목 속도(px/s) + (보정 시) m/s, km/h, mph, 7i 기준 클럽 속도.
    """
    W = get_xy_cols_2d(df, wrist)
    RA = get_xy_cols_2d(df, 'RAnkle')
    LA = get_xy_cols_2d(df, 'LAnkle')

    v_px_s, unit = speed_2d(W, fps)

    impact = detect_impact_by_crossing(df, prefer_2d=True, skip_ratio=0.0, smooth_window=3, hold_frames=0, margin=0.0)

    lo = max(0, impact - 2)
    hi = min(len(v_px_s) - 1, impact + 2)
    peak_local_idx = lo + int(np.nanargmax(v_px_s[lo:hi+1])) if hi >= lo else int(np.nanargmax(v_px_s))
    peak_wrist_px_s = float(v_px_s[peak_local_idx]) if not np.isnan(v_px_s[peak_local_idx]) else float(np.nanmax(v_px_s))

    if m_per_px is not None and m_per_px > 0:
        v_m_s = v_px_s * float(m_per_px)
        v_ms, v_kmh, v_mph = _speed_conversions_m_s(v_m_s)
        peak_wrist_kmh = float(v_kmh[peak_local_idx]) if not np.isnan(v_kmh[peak_local_idx]) else float(np.nanmax(v_kmh))
        peak_wrist_mph = peak_wrist_kmh / 1.609344

        shoulder_width_m = _estimate_shoulder_width_m(df, m_per_px)
        club_info = estimate_club_speed_7i_from_wrist(
            wrist_peak_kmh=peak_wrist_kmh,
            wrist_peak_mph=peak_wrist_mph,
            shoulder_width_m=shoulder_width_m,
        )

        club_kmh = club_info["club_speed_km_h"]
        club_mph = club_info["club_speed_mph"]
        club_kmh_min, club_kmh_max = club_info["club_speed_km_h_range"]
        club_mph_min, club_mph_max = club_info["club_speed_mph_range"]
        k = club_info["club_k_factor"]

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
            'club_k_factor': float(k),
            'club_kmh': float(club_kmh),
            'club_mph': float(club_mph),
            'club_kmh_range': (float(club_kmh_min), float(club_kmh_max)),
            'club_mph_range': (float(club_mph_min), float(club_mph_max)),
            'unit': 'px/s',
            'calibrated_m_per_px': float(m_per_px),
        }

    # 보정 불가 시
    return {
        'impact_frame': int(impact),
        'peak_frame': int(peak_local_idx),
        'v_px_s': v_px_s,
        'wrist_peak_px_s': peak_wrist_px_s,
        'unit': unit,
        'calibrated_m_per_px': None,
        'club_k_factor': None,
        'club_kmh': None,
        'club_mph': None,
        'club_kmh_range': (None, None),
        'club_mph_range': (None, None),
    }

# =========================================================
# 머리 속도 카테고리 (조언용 멘트)
# =========================================================
def categorize_head_speed_mph(head_mph: float):
    refs = [
        ("Female Amateur", 78),
        ("Male Amateur", 93),
        ("LPGA Tour Pro", 94),
        ("PGA Tour Pro (avg male pro)", 114),
        ("Long Driver", 135),
        ("World Record", 157),
    ]
    best = min(refs, key=lambda kv: abs(head_mph - kv[1]))
    name, ref = best
    diff = head_mph - ref
    direction = "빠름" if diff >= 0 else "느림"
    return f"현재 추정 클럽 헤드 속도는 '{name}' 평균 {ref:.0f} mph와 가장 가깝습니다 (Δ{abs(diff):.1f} mph {direction})."

# =========================================================
# 기타 유틸
# =========================================================
def load_cfg(p: Path):
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    raise ValueError("Use YAML for analyze config.")

# =========================================================
# Swing 관련 2D overlay util (기존 구조 유지)
# =========================================================
def get_swing_joints_2d(df: pd.DataFrame, wrist_r: str, wrist_l: str):
    swing_joints = [wrist_l, wrist_r]
    additional_joints = ["LShoulder", "RShoulder", "LElbow", "RElbow"]
    cols_map = parse_joint_axis_map_from_columns(df.columns, prefer_2d=True)
    for joint in additional_joints:
        axes = cols_map.get(joint, {})
        if 'x' in axes and 'y' in axes:
            swing_joints.append(joint)
    print(f"🔗 Swing 관련 관절: {swing_joints}")
    return swing_joints


def build_swing_edges(kp_names):
    E, have = [], set(kp_names)
    def add(a, b):
        if a in have and b in have:
            E.append((a, b))
    add("LShoulder", "LElbow"); add("LElbow", "LWrist")
    add("RShoulder", "RElbow"); add("RElbow", "RWrist")
    add("LShoulder", "RShoulder")
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


def overlay_swing_video(
    img_dir: Path,
    df: pd.DataFrame,
    out_mp4: Path,
    fps: int,
    codec: str,
    wrist_r: str,
    wrist_l: str,
):
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

    kp_names = get_swing_joints_2d(df, wrist_r, wrist_l)
    edges = build_swing_edges(kp_names)
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

    grip_trail = []

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
        print(f"⚠️ 프레임 개수 불일치(swing): images={n_img}, overlay_rows={n_df}.")

    try:
        from runner_utils import is_valid_overlay_point
    except Exception:
        is_valid_overlay_point = None

    for i, p in enumerate(images):
        frame = cv2.imread(p)
        row_idx = i if i < n_df else (n_df - 1 if n_df > 0 else -1)
        row = df.iloc[row_idx] if row_idx >= 0 else None

        for a, b in edges:
            ax, ay, _ = get_xyc_row(row, a)
            bx, by, _ = get_xyc_row(row, b)
            ax, ay = scale_xy(ax, ay)
            bx, by = scale_xy(bx, by)

            if is_valid_overlay_point is not None:
                valid_ab = is_valid_overlay_point(ax, ay, w, h) and is_valid_overlay_point(bx, by, w, h)
            else:
                valid_ab = not (np.isnan(ax) or np.isnan(ay) or np.isnan(bx) or np.isnan(by))

            if valid_ab:
                thickness = 4 if (a == wrist_l and b == wrist_r) else 2
                color = (0, 255, 0) if (a == wrist_l and b == wrist_r) else (0, 255, 255)
                cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), color, thickness)

        for name in kp_names:
            x, y, _ = get_xyc_row(row, name)
            x, y = scale_xy(x, y)
            if is_valid_overlay_point is not None:
                valid_pt = is_valid_overlay_point(x, y, w, h)
            else:
                valid_pt = not (np.isnan(x) or np.isnan(y))
            if valid_pt:
                if name in [wrist_l, wrist_r]:
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)
                else:
                    cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 0), -1)

        lx, ly, _ = get_xyc_row(row, wrist_l)
        rx, ry, _ = get_xyc_row(row, wrist_r)
        lx, ly = scale_xy(lx, ly)
        rx, ry = scale_xy(rx, ry)
        if is_valid_overlay_point is not None:
            valid_grip = is_valid_overlay_point(lx, ly, w, h) and is_valid_overlay_point(rx, ry, w, h)
        else:
            valid_grip = not (np.isnan(lx) or np.isnan(ly) or np.isnan(rx) or np.isnan(ry))

        if valid_grip:
            grip_x = (lx + rx) / 2.0
            grip_y = (ly + ry) / 2.0
            pts = np.array([
                [int(grip_x), int(grip_y-10)],
                [int(grip_x+10), int(grip_y)],
                [int(grip_x), int(grip_y+10)],
                [int(grip_x-10), int(grip_y)]
            ], np.int32)
            cv2.fillPoly(frame, [pts], (0, 255, 0))
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)

            grip_trail.append((int(grip_x), int(grip_y)))
            if len(grip_trail) > 50:
                grip_trail.pop(0)
            for j in range(1, len(grip_trail)):
                a = j / len(grip_trail)
                color_intensity = int(255 * a)
                cv2.line(frame, grip_trail[j-1], grip_trail[j], (color_intensity, 255, 0), 2)

        writer.write(frame)

    writer.release()

    try:
        from runner_utils import transcode_mp4_to_h264, ensure_mp4_faststart
    except Exception:
        transcode_mp4_to_h264 = lambda p, **kw: False
        ensure_mp4_faststart = lambda p: False

    try:
        ok = transcode_mp4_to_h264(str(out_mp4))
        if not ok:
            ensure_mp4_faststart(str(out_mp4))
    except Exception:
        pass

# =========================================================
# run_from_context (백엔드에서 호출)
# =========================================================
def run_from_context(ctx: dict):
    """
    Programmatic runner for swing_speed module (2D/3D 자동 분기).
    """
    try:
        dest = Path(ctx.get('dest_dir', '.'))
        job_id = str(ctx.get('job_id', ctx.get('job', 'job')))
        fps_ctx = ctx.get('fps')
        if fps_ctx is None:
            print(f"[WARN] swing_speed.run_from_context: fps not provided, using fallback 30")
            fps = 30
        else:
            fps = int(fps_ctx)

        wide3 = ctx.get('wide3')
        wide2 = ctx.get('wide2')
        if wide2 is None and wide3 is not None:
            wide2 = wide3
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
                    ctx_for_scale = dict(ctx)
                    ctx_for_scale['wide3'] = use_df
                    if 'intrinsics' in ctx and isinstance(ctx['intrinsics'], dict):
                        ctx_for_scale['intrinsics'] = ctx['intrinsics']
                    scale_to_m = _coord_scale_to_m(ctx_for_scale)
                    anal = analyze_wrist_speed_3d(use_df, fps=fps, wrist=wrist_r, scale_to_m=scale_to_m)
                    N = len(anal['v_m_s'])
                    metrics_df = pd.DataFrame({
                        'frame': range(N),
                        'wrist_speed_m_s': anal['v_m_s'],
                        'wrist_speed_km_h': anal['v_km_h'],
                        'wrist_speed_mph': anal['v_mph'],
                    })
                    summary = {
                        'impact_frame': int(anal['impact_frame']),
                        'peak_frame': int(anal['peak_frame']),
                        'wrist_peak_km_h': float(anal['wrist_peak_kmh']),
                        'wrist_peak_mph': float(anal['wrist_peak_mph']),
                        'club_k_factor': float(anal['club_k_factor']),
                        'club_speed_km_h': float(anal['club_kmh']),
                        'club_speed_mph': float(anal['club_mph']),
                        'club_speed_km_h_range': [float(anal['club_kmh_range'][0]), float(anal['club_kmh_range'][1])],
                        'club_speed_mph_range': [float(anal['club_mph_range'][0]), float(anal['club_mph_range'][1])],
                    }
                else:
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
                        summary = {
                            'impact_frame': int(anal['impact_frame']),
                            'peak_frame': int(anal['peak_frame']),
                            'wrist_peak_km_h': float(anal['wrist_peak_kmh']),
                            'wrist_peak_mph': float(anal['wrist_peak_mph']),
                            'club_k_factor': float(anal['club_k_factor']),
                            'club_speed_km_h': float(anal['club_kmh']),
                            'club_speed_mph': float(anal['club_mph']),
                            'club_speed_km_h_range': [float(anal['club_kmh_range'][0]), float(anal['club_kmh_range'][1])],
                            'club_speed_mph_range': [float(anal['club_mph_range'][0]), float(anal['club_mph_range'][1])],
                            'calibrated_m_per_px': float(anal['calibrated_m_per_px']),
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
                            'club_k_factor': None,
                            'club_speed_km_h': None,
                            'club_speed_mph': None,
                            'club_speed_km_h_range': [None, None],
                            'club_speed_mph_range': [None, None],
                            'calibrated_m_per_px': None,
                        }

                metrics_csv = dest / f"{job_id}_swing_speed_metrics.csv"
                ensure_dir(metrics_csv.parent)
                metrics_df.to_csv(metrics_csv, index=False)
                out['metrics_csv'] = str(metrics_csv)
                out['summary'] = summary
            except Exception as e:
                out['errors']['metrics'] = str(e)
        else:
            out['errors']['metrics'] = 'No DataFrame provided.'

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

        try:
            job_id_local = job_id
            out_json = Path(dest) / "swing_speed_metric_result.json"
            frames_obj = {}
            dimension = out.get('dimension') or ('3d' if out.get('summary') and out.get('summary').get('wrist_peak_km_h') is not None else '2d')

            if dimension == '3d' and 'summary' in out and use_df is not None:
                anal_local = analyze_wrist_speed_3d(use_df, fps=fps, wrist=wrist_r, scale_to_m=_coord_scale_to_m({'wide3': use_df}))
                N = len(anal_local.get('v_m_s', []))
                for i in range(N):
                    vm = float(anal_local['v_m_s'][i]) if np.isfinite(anal_local['v_m_s'][i]) else None
                    vk = float(anal_local['v_km_h'][i]) if np.isfinite(anal_local['v_km_h'][i]) else None
                    vp = float(anal_local['v_mph'][i]) if np.isfinite(anal_local['v_mph'][i]) else None
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
                # 2D 쪽은 간단하게 summary만 포함
                summary = out.get('summary', {})
                out_obj = {
                    'job_id': job_id_local,
                    'dimension': dimension,
                    'overlay_mp4': out.get('overlay_mp4'),
                    'metrics': {
                        'swing_speed': {
                            'summary': summary,
                            'metrics_data': {}
                        }
                    }
                }

            out_json.write_text(__import__('json').dumps(out_obj, ensure_ascii=False, indent=2), encoding='utf-8')
            return out_obj
        except Exception:
            return out

    except Exception as e:
        return {'error': str(e)}

# =========================================================
# CLI main (analyze.yaml 기반)
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Swing Speed 전용 분석기")
    ap.add_argument("-c", "--config", default=str(Path(__file__).parent.parent / "config" / "analyze.yaml"))
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))

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

    lm_cfg = cfg.get("landmarks", {}) or {}
    wrist_l = lm_cfg.get("wrist_left", "LWrist")
    wrist_r = lm_cfg.get("wrist_right", "RWrist")

    out_csv = Path(cfg["metrics_csv"]).parent / "swing_speed_metrics.csv"
    out_mp4 = Path(cfg["overlay_mp4"]).parent / "swing_speed_analysis.mp4"

    df_metrics = None
    df_overlay = None
    if metrics_csv is not None and metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)
        print(f"📋 Metrics CSV 로드(swing): {metrics_csv} ({len(df_metrics)} frames)")
    if overlay_csv is not None and overlay_csv.exists():
        df_overlay = pd.read_csv(overlay_csv)
        print(f"📋 Overlay CSV 로드(swing): {overlay_csv} ({len(df_overlay)} frames)")
    if df_metrics is None and df_overlay is not None:
        print("ℹ️ metrics CSV 없음 → overlay CSV를 metrics 용도로도 사용합니다.")
        df_metrics = df_overlay
    if df_overlay is None and df_metrics is not None:
        print("ℹ️ overlay CSV 없음 → metrics CSV를 overlay 용도로도 사용합니다.")
        df_overlay = df_metrics
    if df_metrics is None or df_overlay is None:
        raise RuntimeError("metrics/overlay CSV를 로드할 수 없습니다. analyze.yaml을 확인하세요.")

    wrist_name = wrist_r
    dim = "3d" if is_dataframe_3d(df_metrics) else "2d"

    if dim == "3d":
        scale_to_m = _coord_scale_to_m({'wide3': df_metrics, **cfg})
        print(f"🧭 좌표 단위 스케일: scale_to_m={scale_to_m:.6f} (m 기준)")
        anal3d = analyze_wrist_speed_3d(df_metrics, fps=fps, wrist=wrist_name, scale_to_m=scale_to_m)
    else:
        m_per_px = _get_m_per_px_from_cfg(cfg, df_overlay)
        if m_per_px is not None:
            print(f"🧭 2D 보정 사용: m_per_px={m_per_px:.6f} → px/s → m/s 변환")
        else:
            print("ℹ️ 2D 보정 스케일이 없어 px/s 단위로만 분석합니다.")
        anal2d = analyze_wrist_speed_2d(df_overlay, fps=fps, wrist=wrist_name, m_per_px=m_per_px)

    job_id = cfg.get("job_id")
    out_dir = Path(cfg.get("metrics_csv", metrics_csv)).parent
    ensure_dir(out_dir)
    out_json = out_dir / "swing_speed_metric_result.json"

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

    if dim == "3d":
        wrist_peak_mph = anal3d['wrist_peak_mph']
        wrist_peak_kmh = anal3d['wrist_peak_kmh']
        club_mph = anal3d['club_mph']
        club_kmh = anal3d['club_kmh']
        club_mph_min, club_mph_max = anal3d['club_mph_range']
        club_kmh_min, club_kmh_max = anal3d['club_kmh_range']
        k_factor = anal3d['club_k_factor']
        advice = categorize_head_speed_mph(club_mph)

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
                        "club_k_factor": float(k_factor),
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
        wrist_peak_px_s = anal2d['wrist_peak_px_s']
        N = len(anal2d['v_px_s'])
        frames_obj = {}
        if anal2d.get('calibrated_m_per_px'):
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
            k_factor = anal2d['club_k_factor']
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
                            "club_k_factor": float(k_factor),
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
                            "club_k_factor": None,
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

    print("\n결과")
    if dim == "3d":
        print(f"실제 swing speed (손목) : {wrist_peak_kmh:.1f} km/h ({wrist_peak_mph:.1f} mph)")
        print(f"추정 club speed (클럽) : {club_kmh:.1f} km/h ({club_mph:.1f} mph)  [k={k_factor:.2f}, 범위 {club_kmh_min:.1f}~{club_kmh_max:.1f} km/h]")
        print(f"📝 조언: {advice}")
    else:
        if anal2d.get('calibrated_m_per_px'):
            print(f"실제 swing speed (손목) : {wrist_peak_kmh:.1f} km/h ({wrist_peak_mph:.1f} mph) [2D 보정]  (m_per_px={anal2d['calibrated_m_per_px']:.6f})")
            print(f"추정 club speed (클럽) : {club_kmh:.1f} km/h ({club_mph:.1f} mph)  [k={k_factor:.2f}, 범위 {club_kmh_min:.1f}~{club_kmh_max:.1f} km/h]")
            print(f"📝 조언: {advice}")
        else:
            print(f"실제 swing speed (손목) : {wrist_peak_px_s:.1f} px/s (2D, 보정 없음)")

if __name__ == "__main__":
    main()
