from typing import List, Optional
import numpy as np
import pandas as pd


def interpolate_sequence(frames_keypoints: List[List[List[float]]], conf_thresh: float = 0.0,
                         method: str = 'linear', fill_method: str = 'none', limit: Optional[int] = None):
    """
    Interpolate skeleton keypoints across frames.
    
    CRITICAL RULE: 절대로 원본 데이터를 필터링으로 날리지 않음
    - 오직 (0,0,0) sentinel만 NaN으로 표시
    - 모든 다른 데이터는 그대로 보존 (confidence 상관없이)
    - NaN이 있는 경우에만 보간 수행
    
    Parameters
    ----------
    frames_keypoints : List[List[List[float]]]
        list of frames; each frame is a list of joints; each joint is [x,y,c]
    conf_thresh : float
        IGNORED - confidence filtering is disabled to preserve all original data
    method : str
        interpolation method ('linear', etc.)
    fill_method : str
        IGNORED - mandatory fill strategy is always applied
    limit : Optional[int]
        interpolation limit
    
    Returns
    -------
    List[List[List[float]]]
        Interpolated frames with NaN only for (0,0,0) sentinel values.
        All other original data preserved.
    """
    if not frames_keypoints:
        return []

    n_frames = len(frames_keypoints)
    # assume consistent joint count; use max joints across frames
    n_joints = max(len(p) for p in frames_keypoints)

    # Build array (frames, joints*3)
    arr = np.full((n_frames, n_joints * 3), np.nan, dtype=float)
    for t, person in enumerate(frames_keypoints):
        for j, kp in enumerate(person):
            try:
                x = float(kp[0])
                y = float(kp[1])
                c = float(kp[2])
            except Exception:
                x, y, c = np.nan, np.nan, np.nan
            arr[t, j*3 + 0] = x
            arr[t, j*3 + 1] = y
            arr[t, j*3 + 2] = c

    xcols = arr[:, 0::3]
    ycols = arr[:, 1::3]
    ccols = arr[:, 2::3]
    
    # === ONLY FILTERING: Mark (0,0,0) sentinel as NaN ===
    # 이것만 필터링 - OpenPose는 missing detection을 (0,0,0)으로 표시
    sentinel = (xcols == 0.0) & (ycols == 0.0) & (ccols == 0.0)
    for j in range(n_joints):
        mask = sentinel[:, j]
        arr[mask, j*3:(j+1)*3] = np.nan
    
    # === NO OTHER FILTERING ===
    # conf_thresh parameter is IGNORED - no confidence filtering
    # All other data is preserved as-is
    
    # Convert to DataFrame and interpolate column-wise
    cols = []
    for j in range(n_joints):
        cols += [f'x_{j}', f'y_{j}', f'c_{j}']
    df = pd.DataFrame(arr, columns=cols)

    # Interpolate only NaN values (sentinel-marked only)
    df_interp = df.interpolate(method=method, axis=0, limit=limit, limit_direction='both')

    # === MANDATORY FILL for remaining NaN ===
    # Only applies to NaN that couldn't be interpolated
    # ffill → bfill → zero fill
    df_interp = df_interp.ffill(limit=None)
    df_interp = df_interp.bfill(limit=None)
    df_interp = df_interp.fillna(0.0)
    df_interp = df_interp.replace([np.inf, -np.inf], 0.0)

    # Reconstruct frames
    out = []
    darr = df_interp.values
    for t in range(n_frames):
        person = []
        for j in range(n_joints):
            x = darr[t, j*3 + 0]
            y = darr[t, j*3 + 1]
            c = darr[t, j*3 + 2]
            # Convert non-finite to 0.0 for output safety
            if not np.isfinite(x):
                x = 0.0
            if not np.isfinite(y):
                y = 0.0
            if not np.isfinite(c):
                c = 0.0
            person.append([float(x), float(y), float(c)])
        out.append(person)

    return out
