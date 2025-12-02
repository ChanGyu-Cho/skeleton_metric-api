"""
impact_utils.py
공용 임팩트 프레임 계산 유틸리티

모듈 간 일관성을 위해 다음 기능을 제공합니다:
- compute_stance_mid_and_width(df, prefer_2d): 좌우 발목으로 스탠스 중앙/폭 계산
- detect_impact_by_crossing(df, prefer_2d, skip_ratio, smooth_window, hold_frames, margin):
    RWrist X가 스탠스 중앙을 +방향으로 교차하는 첫 프레임을 임팩트로 탐지

주의: 본 유틸은 최소 의존으로 설계되어 각 모듈의 parse 함수를 사용하지 않습니다.
      내부에서 간단한 컬럼 매핑(parse_joint_axis_map_from_columns)을 포함합니다.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def parse_joint_axis_map_from_columns(columns, prefer_2d: bool = False) -> Dict[str, Dict[str, str]]:
    """주어진 컬럼 리스트에서 관절명과 axis 컬럼명을 매핑합니다.

    반환값 예시: {'Nose': {'x':'Nose__x','y':'Nose__y','z':'Nose__z'}, ...}

    지원하는 패턴:
      - Joint__x, Joint__y, Joint__z
      - Joint_X3D, Joint_Y3D, Joint_Z3D
      - Joint_X, Joint_Y, Joint_Z
      - Joint_x, Joint_y, Joint_z (2D preferred)
    """
    cols = list(columns)
    mapping: Dict[str, Dict[str, str]] = {}

    # 후보 패턴을 나열 (우선순위가 높은 것부터)
    if prefer_2d:
        # 2D 좌표 우선 (소문자 _x/_y), 그 다음 일반/3D 변형
        axis_patterns = [
            ('_x', '_y', '_z'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_X3D', '_Y3D', '_Z3D'),
        ]
    else:
        # 3D 좌표 우선 (X3D), 그 다음 일반/2D
        axis_patterns = [
            ('_X3D', '_Y3D', '_Z3D'),
            ('__x', '__y', '__z'),
            ('_X', '_Y', '_Z'),
            ('_x', '_y', '_z'),
        ]

    # 빠른 탐색을 위해 컬럼 세트를 준비
    col_set = set(cols)

    # 시도: 각 컬럼을 기준으로 관절명을 추정
    for col in cols:
        # skip columns that clearly aren't joints (e.g., frame, time)
        if isinstance(col, str) and col.lower() in ('frame', 'time', 'timestamp'):
            continue
        for x_pat, y_pat, z_pat in axis_patterns:
            if isinstance(col, str) and col.endswith(x_pat):
                joint = col[:-len(x_pat)]
                x_col = joint + x_pat
                y_col = joint + y_pat
                z_col = joint + z_pat
                if x_col in col_set and y_col in col_set:
                    # z may be missing for 2D datasets
                    mapping.setdefault(joint, {})['x'] = x_col
                    mapping.setdefault(joint, {})['y'] = y_col
                    if z_col in col_set:
                        mapping[joint]['z'] = z_col
                    break

    return mapping


def _get_axis_series(df: pd.DataFrame, joint: str, axis: str, prefer_2d: bool) -> pd.Series:
    """특정 관절의 축 시계열 추출"""
    cmap = parse_joint_axis_map_from_columns(df.columns, prefer_2d=prefer_2d)
    col = (cmap.get(joint, {}) or {}).get(axis)
    if col and col in df.columns:
        try:
            return df[col].astype(float)
        except Exception:
            return pd.to_numeric(df[col], errors='coerce')
    return pd.Series([np.nan] * len(df), index=df.index, dtype=float)


def compute_stance_mid_and_width(df: pd.DataFrame, prefer_2d: bool = False) -> Tuple[np.ndarray, float]:
    """프레임별 스탠스 중앙과 스탠스 폭(중앙값)을 계산합니다.

    prefer_2d=True이면 2D 헤더 우선으로 탐색합니다.
    반환: (stance_mid_x[n], stance_width_med)
    """
    ra_x = _get_axis_series(df, 'RAnkle', 'x', prefer_2d=prefer_2d).to_numpy()
    la_x = _get_axis_series(df, 'LAnkle', 'x', prefer_2d=prefer_2d).to_numpy()
    stance_mid = (ra_x + la_x) / 2.0
    width_inst = np.abs(ra_x - la_x)
    width_vals = width_inst[~np.isnan(width_inst)]
    stance_width_med = float(np.median(width_vals)) if width_vals.size > 0 else np.nan
    return stance_mid, stance_width_med


def detect_impact_by_crossing(
    df: pd.DataFrame,
    prefer_2d: bool = False,
    skip_ratio: float = 0.0,
    smooth_window: int = 3,
    hold_frames: int = 0,
    margin: float = 0.0,
    wrist_prefer: str = 'R',
    fps: Optional[int] = None,
) -> int:
    """COM 방식(스탠스 중앙 교차) 임팩트 탐지.

    규칙:
      - 시작 프레임: floor(N*skip_ratio) 또는 fps 기반 정규화 이후부터 탐색
      - 손목 선택: wrist_prefer ('R' 또는 'L'), 기본 R
      - 조건: wrist[i-1] < mid[i-1]+margin and wrist[i] >= mid[i]+margin and Δwrist>0
      - smooth_window: 스무딩 윈도우 크기 (median->moving 조합)
      - hold_frames>0 이면 조건 충족 이후 연속 유지 검증
      - 미검출 시 fallback: wrist_x 최대 프레임

    Args:
        df: 관절 좌표 데이터프레임
        prefer_2d: 2D 컬럼 우선 탐색 여부
        skip_ratio: 초반 건너뛸 구간 비율 (0.0~1.0) - fps가 None이면 N*skip_ratio, 있으면 fps 기반 정규화
        smooth_window: 스무딩 윈도우 크기
        hold_frames: 임팩트 후 유지 프레임 수
        margin: 스탠스 중앙 기준 마진
        wrist_prefer: 'R' 또는 'L' 손목 선택
        fps: 프레임 레이트 (있으면 skip_ratio를 프레임 수로 정규화)

    Returns:
        임팩트 프레임 인덱스 (찾지 못하면 최대 X 프레임)
    """
    N = len(df)
    if N == 0:
        return -1
    
    stance_mid, _ = compute_stance_mid_and_width(df, prefer_2d=prefer_2d)
    rw = _get_axis_series(df, 'RWrist', 'x', prefer_2d=prefer_2d).to_numpy()
    lw = _get_axis_series(df, 'LWrist', 'x', prefer_2d=prefer_2d).to_numpy()
    wrist_x = rw if (wrist_prefer.upper() == 'R') else lw

    # 간단 스무딩으로 작은 진동 완화 (median->moving)
    def _smooth(a: np.ndarray, w: int = 3) -> np.ndarray:
        s = pd.Series(a).interpolate(limit_direction='both')
        med = s.rolling(w, center=True, min_periods=1).median()
        mov = med.rolling(w, center=True, min_periods=1).mean()
        return mov.to_numpy()

    sm_wrist = _smooth(wrist_x, w=int(max(1, smooth_window)))
    sm_mid = _smooth(stance_mid, w=int(max(1, smooth_window)))

    # fps 기반 skip_ratio 정규화: skip_ratio는 여전히 0~1.0 값으로 입력받고,
    # 필요하면 프레임 수로 변환 (예: 60fps에서 0.3 = 18 프레임)
    if fps is not None and fps > 0:
        # skip_ratio가 시간(초) 기반이 아니라 비율이므로 그대로 사용
        # 단, fps를 사용해 임팩트 감지 로직을 조정할 수 있음 (향후 확장)
        pass
    
    start = int(max(0, np.floor(N * float(max(0.0, min(1.0, skip_ratio))))))
    impact = -1
    for i in range(max(1, start), N):
        if np.isnan(sm_wrist[i]) or np.isnan(sm_wrist[i-1]) or np.isnan(sm_mid[i]) or np.isnan(sm_mid[i-1]):
            continue
        crossed = (sm_wrist[i-1] < sm_mid[i-1] + margin) and (sm_wrist[i] >= sm_mid[i] + margin)
        positive_dx = (sm_wrist[i] - sm_wrist[i-1]) > 0
        if crossed and positive_dx:
            ok = True
            if hold_frames > 0:
                for k in range(1, int(max(0, hold_frames))):
                    j = i + k
                    if j >= N or np.isnan(sm_wrist[j]) or np.isnan(sm_mid[j]) or (sm_wrist[j] < sm_mid[j] + margin):
                        ok = False
                        break
            if ok:
                impact = i
                break

    if impact == -1:
        with np.errstate(invalid='ignore'):
            impact = int(np.nanargmax(wrist_x)) if np.any(~np.isnan(wrist_x)) else (N - 1)
    return int(impact)


def detect_impact_by_wrist_crossing(wrist_x: np.ndarray, stance_mid_x: np.ndarray, fps: Optional[int] = None) -> int:
    """단순 손목-스탠스 교차 기반 임팩트 탐지 (swing_speed.py 호환)
    
    X 증가(+) 방향으로 스탠스 중앙을 넘는 첫 프레임을 임팩트로 탐지
    
    Args:
        wrist_x: 손목 X 좌표 배열
        stance_mid_x: 스탠스 중앙 X 좌표 배열
        fps: 프레임 레이트 (미래 확장용, 현재 미사용)
    
    Returns:
        임팩트 프레임 인덱스
    """
    N = len(wrist_x)
    impact = -1
    for i in range(1, N):
        if np.isnan(wrist_x[i]) or np.isnan(wrist_x[i-1]) or np.isnan(stance_mid_x[i]) or np.isnan(stance_mid_x[i-1]):
            continue
        crossed = (wrist_x[i-1] < stance_mid_x[i-1]) and (wrist_x[i] >= stance_mid_x[i])
        positive_dx = (wrist_x[i] - wrist_x[i-1]) > 0
        if crossed and positive_dx:
            impact = i
            break
    if impact == -1:
        with np.errstate(invalid='ignore'):
            impact = int(np.nanargmax(wrist_x)) if np.any(~np.isnan(wrist_x)) else N-1
    return impact


def _median_ignore_nan(arr: np.ndarray) -> float:
    """NaN을 무시하고 중앙값 계산"""
    arr = np.asarray(arr, dtype=float)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.nan
    return float(np.median(valid))


def _get_axis_series_helper(df: pd.DataFrame, joint: str, axis: str, prefer_2d: bool) -> np.ndarray:
    """특정 관절의 축 시계열 추출 (numpy 배열로 반환)"""
    s = _get_axis_series(df, joint, axis, prefer_2d)
    return s.to_numpy(dtype=float)


def detect_impact_with_wrist_selection(
    df: pd.DataFrame,
    prefer_2d: bool = False,
    skip_ratio: float = 0.2,
    fps: Optional[int] = None,
) -> Tuple[int, str, np.ndarray]:
    """손목 선택과 함께 임팩트 탐지 (head_speed.py 호환)
    
    손목 선택은 후반부 기울기 기반으로 이루어집니다.
    
    Args:
        df: 관절 좌표 데이터프레임
        prefer_2d: 2D 컬럼 우선 탐색 여부
        skip_ratio: 초반 건너뛸 구간 비율 (기울기 계산용)
        fps: 프레임 레이트 (미래 확장용, 현재 미사용)
    
    Returns:
        (impact_frame, selected_wrist, stance_mid_x): 임팩트 프레임, 선택된 손목, 스탠스 중앙값
    """
    N = len(df)
    if N == 0:
        return -1, 'RWrist', np.array([])
    
    # 좌표 시계열 추출
    la_x = _get_axis_series_helper(df, 'LAnkle', 'x', prefer_2d)
    ra_x = _get_axis_series_helper(df, 'RAnkle', 'x', prefer_2d)
    lw_x = _get_axis_series_helper(df, 'LWrist', 'x', prefer_2d)
    rw_x = _get_axis_series_helper(df, 'RWrist', 'x', prefer_2d)
    
    # 스탠스 중앙
    stance_mid_x = (ra_x + la_x) / 2.0
    
    # 손목 선택: 후반부(skip_ratio 이후) 구간에서 선형 기울기 비교
    start_slope = int(N * max(skip_ratio, 0.2))
    start_slope = min(start_slope, max(N - 3, 0))
    xs = np.arange(start_slope, N, dtype=float)
    
    def slope_of(arr: np.ndarray) -> float:
        """배열 구간의 선형 기울기 계산"""
        yy = arr[start_slope:]
        if len(xs) != len(yy) or len(yy) < 2:
            return np.nan
        # NaN 처리: 보간 후 회귀
        yy2 = pd.Series(yy).interpolate(limit_direction='both').to_numpy()
        try:
            k, b = np.polyfit(xs, yy2, 1)
            return float(k)
        except Exception:
            return np.nan
    
    slope_L = slope_of(lw_x)
    slope_R = slope_of(rw_x)
    selected_wrist = 'RWrist' if (np.nan_to_num(slope_R, nan=-1e9) >= np.nan_to_num(slope_L, nan=-1e9)) else 'LWrist'
    wrist_x = rw_x if selected_wrist == 'RWrist' else lw_x
    
    # 임팩트 탐지: skip_ratio 이후 구간에서
    start = int(N * max(skip_ratio, 0.2))
    impact = -1
    for i in range(max(1, start), N):
        if np.isnan(wrist_x[i]) or np.isnan(wrist_x[i-1]) or np.isnan(stance_mid_x[i]):
            continue
        cond_cross = wrist_x[i] >= stance_mid_x[i]
        cond_vel = (wrist_x[i] - wrist_x[i-1]) > 0
        if cond_cross and cond_vel:
            impact = i
            break
    
    if impact == -1:
        # fallback: 손목 X 최대 프레임
        with np.errstate(invalid='ignore'):
            impact = int(np.nanargmax(wrist_x)) if np.any(~np.isnan(wrist_x)) else N - 1
    
    return int(impact), selected_wrist, stance_mid_x
