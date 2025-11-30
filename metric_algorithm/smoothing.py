"""
Smoothing filters for skeleton data (2D/3D coordinates)

주로 OpenPose 노이즈(순간적으로 튀는 좌표)를 제거하기 위해 사용.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from typing import List, Optional, Tuple


def butterworth_filter(data: np.ndarray, order: int = 2, cutoff: float = 0.1, 
                       fps: float = 30.0) -> np.ndarray:
    """
    Butterworth 저역 필터 적용
    
    중요: NaN 값은 보존됨 (필터링 후 재복원)
    
    Parameters
    ----------
    data : np.ndarray
        입력 배열 (1D 또는 2D: [frames, joints*3])
    order : int
        필터 order (기본 2)
    cutoff : float
        정규화된 cutoff 주파수 (0-1, 기본 0.1)
    fps : float
        샘플링 레이트 (기본 30 fps)
    
    Returns
    -------
    np.ndarray
        필터된 데이터 (같은 shape, NaN 패턴 유지)
    """
    if data.size == 0:
        return data
    
    # 1D 데이터 처리
    if data.ndim == 1:
        # NaN 위치 기록 (나중에 복원)
        mask = np.isfinite(data)
        if not np.any(mask):
            return data
        
        # butter SOS 필터 설계
        sos = butter(order, cutoff, output='sos')
        
        # 필터 적용 (NaN 무시)
        x = data.copy()
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            # 보간하여 필터 적용
            valid_data = x[valid_indices]
            filtered = sosfilt(sos, valid_data)
            x[valid_indices] = filtered
            
            # **중요**: 필터링 후 원본 NaN 위치 복원
            x[~mask] = np.nan
        
        return x
    
    # 2D 데이터 처리 (프레임 x 특성)
    elif data.ndim == 2:
        result = data.copy()
        for col_idx in range(data.shape[1]):
            col = data[:, col_idx]
            mask = np.isfinite(col)
            
            if not np.any(mask):
                continue
            
            sos = butter(order, cutoff, output='sos')
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) > 0:
                valid_data = col[valid_indices]
                filtered = sosfilt(sos, valid_data)
                result[valid_indices, col_idx] = filtered
                
                # **중요**: 필터링 후 원본 NaN 위치 복원
                result[~mask, col_idx] = np.nan
        
        return result
    
    return data


def smooth_skeleton_tidy(df: pd.DataFrame, order: int = 2, cutoff: float = 0.1,
                         fps: float = 30.0) -> pd.DataFrame:
    """
    Tidy 포맷 스켈레톤 데이터 평활화
    
    Tidy 포맷: (frame, person_idx, joint_idx, x, y, conf, X, Y, Z)
    
    Parameters
    ----------
    df : pd.DataFrame
        Tidy 포맷 스켈레톤 데이터
    order : int
        Butterworth 필터 order
    cutoff : float
        Cutoff 주파수 (정규화)
    fps : float
        샘플링 레이트
    
    Returns
    -------
    pd.DataFrame
        평활화된 데이터
    """
    if df.empty:
        return df
    
    df_smooth = df.copy()
    
    # person별로 그룹화하여 필터 적용
    for person_idx in df['person_idx'].unique():
        mask = df['person_idx'] == person_idx
        person_data = df[mask]
        
        # joint별로 필터 적용
        for joint_idx in person_data['joint_idx'].unique():
            joint_mask = person_data['joint_idx'] == joint_idx
            joint_indices = df[(df['person_idx'] == person_idx) & 
                              (df['joint_idx'] == joint_idx)].index
            
            if len(joint_indices) == 0:
                continue
            
            # 2D 좌표 (x, y) 필터
            for col in ['x', 'y']:
                if col in df.columns:
                    col_data = df.loc[joint_indices, col].values
                    col_filtered = butterworth_filter(col_data, order, cutoff, fps)
                    df_smooth.loc[joint_indices, col] = col_filtered
            
            # 3D 좌표 (X, Y, Z) 필터 (있으면)
            for col in ['X', 'Y', 'Z']:
                if col in df.columns:
                    col_data = df.loc[joint_indices, col].values
                    col_filtered = butterworth_filter(col_data, order, cutoff, fps)
                    df_smooth.loc[joint_indices, col] = col_filtered
    
    return df_smooth


def smooth_skeleton_wide(df: pd.DataFrame, order: int = 2, cutoff: float = 0.1,
                         fps: float = 30.0, dimension: str = '2d') -> pd.DataFrame:
    """
    Wide 포맷 스켈레톤 데이터 평활화
    
    Wide 포맷: 프레임별 행, 관절별 열 (Nose_x, Nose_y, Nose_c, ...)
    또는 3D: (Nose__x, Nose__y, Nose__z, ...)
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide 포맷 스켈레톤 데이터
    order : int
        Butterworth 필터 order
    cutoff : float
        Cutoff 주파수
    fps : float
        샘플링 레이트
    dimension : str
        '2d' 또는 '3d'
    
    Returns
    -------
    pd.DataFrame
        평활화된 데이터
    """
    if df.empty:
        return df
    
    df_smooth = df.copy()
    
    # 필터링할 컬럼 식별
    if dimension == '2d':
        # _x, _y, _c 패턴
        coord_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    else:  # 3d
        # __x, __y, __z 패턴
        coord_cols = [c for c in df.columns if c.endswith('__x') or c.endswith('__y') or c.endswith('__z')]
    
    for col in coord_cols:
        col_data = df[col].values
        col_filtered = butterworth_filter(col_data, order, cutoff, fps)
        df_smooth[col] = col_filtered
    
    return df_smooth


def remove_jitter(keypoints: List[List[List[float]]], threshold: float = 50.0) -> List[List[List[float]]]:
    """
    순간적 튀는 좌표 제거 (threshold 기반)
    
    연속된 프레임 간 좌표 변화가 threshold를 초과하면 이전값으로 유지
    
    Parameters
    ----------
    keypoints : List[List[List[float]]]
        프레임별 관절 좌표 ([[x, y, c], ...])
    threshold : float
        최대 허용 변화 픽셀수 (기본 50)
    
    Returns
    -------
    List[List[List[float]]]
        jitter 제거된 좌표
    """
    if len(keypoints) < 2:
        return keypoints
    
    result = [keypoints[0].copy()]
    
    for frame_idx in range(1, len(keypoints)):
        current_frame = keypoints[frame_idx].copy()
        prev_frame = result[-1]
        
        # 각 관절별로 변화량 확인
        for joint_idx, (x, y, c) in enumerate(current_frame):
            if joint_idx >= len(prev_frame):
                continue
            
            px, py, pc = prev_frame[joint_idx]
            
            # confidence 확인 (conf > 0.1이어야 유효)
            if c < 0.1 or pc < 0.1:
                continue
            
            # 거리 계산
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            
            # 임계값 초과 시 이전값 사용
            if dist > threshold:
                current_frame[joint_idx] = [px, py, c]
        
        result.append(current_frame)
    
    return result


def calculate_z_bounds(df_3d: pd.DataFrame, n_std: float = 2.5, 
                       min_valid_z: float = 0.1, max_valid_z: float = 5.0,
                       remove_outliers_for_stats: bool = True) -> Tuple[float, float]:
    """
    3D 스켈레톤 데이터에서 동적 Z 범위(bounds) 계산
    
    전체 프레임의 유효한 Z값을 분석하여, 정상 범위를 통계적으로 결정.
    사람의 움직임 특성(몸통 중심에서 크게 벗어나지 않음)을 이용.
    
    Parameters
    ----------
    df_3d : pd.DataFrame
        Wide format 3D 스켈레톤 (frame × joints*3)
        Column format: "{joint_name}__z" (예: "Nose__z", "RWrist__z")
    n_std : float
        기준편차의 배수 (기본 2.5 → ~99% 신뢰도)
    min_valid_z : float
        절대 최소값 (cm) (기본 0.1m, 카메라 오류 제외)
    max_valid_z : float
        절대 최대값 (m) (기본 5.0m, 배경 거리)
    remove_outliers_for_stats : bool
        통계 계산 시 이상치 1차 제거 (기본 True, IQR 사용)
    
    Returns
    -------
    Tuple[float, float]
        (z_min, z_max) - 유효 Z 범위 (meter 단위)
    
    Examples
    --------
    >>> df_3d = pd.read_csv('skeleton3d.csv', index_col=0)
    >>> z_min, z_max = calculate_z_bounds(df_3d, n_std=2.5)
    >>> print(f"Z bounds: {z_min:.2f}m ~ {z_max:.2f}m")
    Z bounds: 1.52m ~ 2.48m
    """
    # Z 컬럼 추출 (__z로 끝나는 컬럼)
    z_cols = [col for col in df_3d.columns if col.endswith('__z')]
    
    if not z_cols:
        # Z 컬럼이 없으면 기본값 반환
        return (min_valid_z, max_valid_z)
    
    # 모든 Z값 수집 (flat)
    z_values = df_3d[z_cols].values.flatten()
    
    # NaN, inf 제거
    z_valid = z_values[np.isfinite(z_values)]
    
    if len(z_valid) == 0:
        # 유효한 값이 없으면 기본값
        return (min_valid_z, max_valid_z)
    
    # 1차 필터: 절대 범위 적용
    z_valid = z_valid[(z_valid >= min_valid_z) & (z_valid <= max_valid_z)]
    
    if len(z_valid) == 0:
        return (min_valid_z, max_valid_z)
    
    # 2차 필터: IQR 기반 이상치 제거 (선택사항)
    if remove_outliers_for_stats and len(z_valid) > 10:
        q1, q3 = np.percentile(z_valid, [25, 75])
        iqr = q3 - q1
        
        # IQR의 1.5배 범위 밖 제거 (표준 이상치 정의)
        z_valid = z_valid[(z_valid >= q1 - 1.5*iqr) & (z_valid <= q3 + 1.5*iqr)]
        
        if len(z_valid) == 0:
            return (min_valid_z, max_valid_z)
    
    # 통계 계산
    mean_z = np.mean(z_valid)
    std_z = np.std(z_valid)
    
    # 동적 범위 계산
    z_min = max(min_valid_z, mean_z - n_std * std_z)
    z_max = min(max_valid_z, mean_z + n_std * std_z)
    
    return (z_min, z_max)


def validate_z_value(z_raw: float, z_bounds: Optional[Tuple[float, float]] = None,
                     fallback_value: Optional[float] = None,
                     depth_scale: float = 0.001) -> Optional[float]:
    """
    단일 Z값 유효성 검증 (깊이 샘플링 직후, X/Y 계산 전에 사용)
    
    Parameters
    ----------
    z_raw : float
        깊이맵에서 샘플된 원본 깊이값 (정규화 전, raw unit)
    z_bounds : Optional[Tuple[float, float]]
        (z_min, z_max) 범위 (meter) - None이면 절대값 사용
    fallback_value : Optional[float]
        범위 밖일 때 사용할 대체값 (meter)
    depth_scale : float
        깊이값 변환 스케일 (기본 0.001, RealSense)
    
    Returns
    -------
    Optional[float]
        유효한 Z값(meter) 또는 None(invalid)
    
    Notes
    -----
    - z_raw=0 또는 np.nan → None (홀 또는 깊이 부재)
    - z_bounds 범위 밖 → fallback_value 또는 None
    - 유효하면 meter 단위로 변환하여 반환
    """
    # 홀/깊이 부재 체크
    if z_raw == 0 or not np.isfinite(float(z_raw)):
        return fallback_value
    
    # meter 단위로 변환
    z_meter = float(z_raw) * depth_scale
    
    # 범위 체크
    if z_bounds is not None:
        z_min, z_max = z_bounds
        if z_meter < z_min or z_meter > z_max:
            return fallback_value
    
    return z_meter


def filter_z_outliers_by_frame_delta(df_tidy: pd.DataFrame, 
                                     joint_threshold: float = 0.4,
                                     depth_scale: float = 0.001) -> pd.DataFrame:
    """
    2단계 필터링: 프레임 간 Z 값 연속성 기반 이상치 제거
    
    같은 관절의 연속 프레임 간 Z 변화가 threshold를 초과하면,
    이전 프레임의 값으로 대체.
    
    Parameters
    ----------
    df_tidy : pd.DataFrame
        Tidy format 3D skeleton (frame, person_idx, joint_idx, x, y, conf, X, Y, Z)
    joint_threshold : float
        프레임 간 허용 Z 변화량 (meter, 기본 0.4m)
    depth_scale : float
        깊이값 스케일 (기본 0.001)
    
    Returns
    -------
    pd.DataFrame
        필터링된 tidy DataFrame
    
    Notes
    -----
    - raw 깊이값 → meter로 변환하여 비교
    - 튄 값은 이전 프레임값으로 대체
    - 대체할 이전값이 없으면 NaN 유지
    """
    df = df_tidy.copy()
    
    # 관절별, 사람별로 그룹화
    for (person_idx, joint_idx), group in df.groupby(['person_idx', 'joint_idx']):
        indices = group.index
        z_values = df.loc[indices, 'Z'].values
        
        # meter 단위로 변환
        z_meter = z_values * depth_scale
        
        # 프레임 간 차이 계산 (유효한 값만)
        valid_mask = np.isfinite(z_meter)
        z_valid_indices = np.where(valid_mask)[0]
        
        if len(z_valid_indices) < 2:
            continue
        
        # 연속 프레임의 Z 차이 계산
        for i in range(1, len(z_valid_indices)):
            prev_idx = z_valid_indices[i - 1]
            curr_idx = z_valid_indices[i]
            
            # 실제 프레임 간 거리 (NaN을 건너뜀)
            z_delta = np.abs(z_meter[curr_idx] - z_meter[prev_idx])
            
            # 큰 변화 감지 → 현재값을 NaN으로 마킹
            if z_delta > joint_threshold:
                df.loc[indices[curr_idx], 'Z'] = np.nan
                df.loc[indices[curr_idx], 'X'] = np.nan
                df.loc[indices[curr_idx], 'Y'] = np.nan
    
    return df


def filter_z_outliers_by_frame_consistency(df_tidy: pd.DataFrame,
                                          body_part_groups: Optional[dict] = None,
                                          consistency_threshold: float = 0.6,
                                          depth_scale: float = 0.001) -> pd.DataFrame:
    """
    3단계 필터링: 프레임 내 Z 값 일관성 기반 이상치 제거
    
    같은 프레임 내 어깨/팔꿈치 등 신체 중심부의 중앙값을 기준으로,
    개별 관절의 Z가 크게 벗어나면 이상치로 판단.
    
    Parameters
    ----------
    df_tidy : pd.DataFrame
        Tidy format 3D skeleton
    body_part_groups : dict
        신체 부위별 관절 인덱스 (예: {'core': [13, 14], 'left_arm': [5, 6, 7]})
    consistency_threshold : float
        중앙값으로부터 허용 편차 (meter, 기본 0.6m)
    depth_scale : float
        깊이값 스케일
    
    Returns
    -------
    pd.DataFrame
        필터링된 tidy DataFrame
    
    Notes
    -----
    - 신체 중심(어깨, 팔꿈치)을 기준으로 일관성 검증
    - COCO-17 기준: Shoulders (11, 12), Elbows (7, 8)
    """
    if body_part_groups is None:
        # COCO-17 기준 신체 부위 (joint_idx)
        body_part_groups = {
            'core': [11, 12],  # LShoulder, RShoulder
            'upper_limbs': [5, 6, 7, 8]  # LElbow, RElbow, LWrist, RWrist
        }
    
    df = df_tidy.copy()
    
    # 프레임별로 일관성 검증
    for frame_idx in df['frame'].unique():
        frame_data = df[df['frame'] == frame_idx]
        
        for person_idx in frame_data['person_idx'].unique():
            person_frame = df[(df['frame'] == frame_idx) & (df['person_idx'] == person_idx)]
            
            # 신체 중심부의 Z 값들 (core joints)
            core_z = person_frame[person_frame['joint_idx'].isin(body_part_groups['core'])]['Z']
            core_z_valid = core_z.dropna() * depth_scale
            
            if len(core_z_valid) < 1:
                continue
            
            # 신체 중심의 중앙값
            core_median = np.median(core_z_valid)
            
            # 다른 관절들의 Z 일관성 검증
            for idx in person_frame.index:
                z_val = df.loc[idx, 'Z']
                if not np.isfinite(z_val):
                    continue
                
                z_meter = z_val * depth_scale
                
                # 중앙값으로부터 편차 계산
                z_deviation = np.abs(z_meter - core_median)
                
                # 편차가 threshold를 초과하면 이상치
                if z_deviation > consistency_threshold:
                    df.loc[idx, 'Z'] = np.nan
                    df.loc[idx, 'X'] = np.nan
                    df.loc[idx, 'Y'] = np.nan
    
    return df
