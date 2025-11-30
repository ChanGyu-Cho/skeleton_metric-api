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
        필터된 데이터 (같은 shape)
    """
    if data.size == 0:
        return data
    
    # 1D 데이터 처리
    if data.ndim == 1:
        # NaN 처리
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
