#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COM 신뢰도 가중치 기능 테스트 스크립트

사용법:
    python test_com_confidence.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# metric_algorithm 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from com_speed import compute_com_points_3d, compute_com_points_2d


def test_confidence_weighting_3d():
    """3D 신뢰도 가중치 테스트"""
    print("\n" + "="*60)
    print("Test 1: 3D 신뢰도 가중치")
    print("="*60)
    
    # 샘플 데이터: 4개 관절
    data = {
        'LShoulder__x': [100.0, 110.0],
        'LShoulder__y': [200.0, 210.0],
        'LShoulder__z': [300.0, 310.0],
        'LShoulder__c': [0.95, 0.90],  # 신뢰도 높음
        
        'RShoulder__x': [110.0, 120.0],
        'RShoulder__y': [210.0, 220.0],
        'RShoulder__z': [310.0, 320.0],
        'RShoulder__c': [0.50, 0.60],  # 신뢰도 낮음
        
        'LHip__x': [90.0, 100.0],
        'LHip__y': [150.0, 160.0],
        'LHip__z': [250.0, 260.0],
        'LHip__c': [0.98, 0.95],  # 신뢰도 매우 높음
        
        'RHip__x': [100.0, 110.0],
        'RHip__y': [160.0, 170.0],
        'RHip__z': [260.0, 270.0],
        'RHip__c': [0.70, 0.75],
    }
    
    df = pd.DataFrame(data)
    
    # 신뢰도 가중치 적용
    print("\n[신뢰도 가중치 사용]")
    com_weighted = compute_com_points_3d(df, use_confidence=True)
    
    # 동일 가중치 적용
    print("\n[동일 가중치 사용]")
    com_equal = compute_com_points_3d(df, use_confidence=False)
    
    print(f"\nFrame 0 결과:")
    print(f"  신뢰도 가중치: {com_weighted[0]}")
    print(f"  동일 가중치:  {com_equal[0]}")
    print(f"  차이: {np.abs(com_weighted[0] - com_equal[0])}")
    
    # 해석
    print(f"\n[해석]")
    print(f"  LShoulder과 LHip는 신뢰도가 높으므로 (0.95, 0.98)")
    print(f"  신뢰도 가중치 모드에서 이들의 영향이 더 큼")
    print(f"  RShoulder는 신뢰도가 낮으므로 (0.50) 영향 감소")
    
    assert not np.allclose(com_weighted, com_equal), "가중치가 다른 결과를 생성해야 함"
    print("✓ 테스트 통과")


def test_confidence_missing_3d():
    """3D 신뢰도 컬럼 없을 때 테스트"""
    print("\n" + "="*60)
    print("Test 2: 3D 신뢰도 컬럼 없음")
    print("="*60)
    
    # 신뢰도 컬럼 없음
    data = {
        'LShoulder__x': [100.0, 110.0],
        'LShoulder__y': [200.0, 210.0],
        'LShoulder__z': [300.0, 310.0],
        
        'RShoulder__x': [110.0, 120.0],
        'RShoulder__y': [210.0, 220.0],
        'RShoulder__z': [310.0, 320.0],
    }
    
    df = pd.DataFrame(data)
    
    print("\n[신뢰도 가중치 지정했으나 컬럼 없음]")
    com_weighted = compute_com_points_3d(df, use_confidence=True)
    
    print("\n[동일 가중치 명시]")
    com_equal = compute_com_points_3d(df, use_confidence=False)
    
    print(f"\nFrame 0 결과:")
    print(f"  신뢰도 가중치: {com_weighted[0]}")
    print(f"  동일 가중치:  {com_equal[0]}")
    
    # 신뢰도 컬럼이 없으면 결과가 같아야 함
    assert np.allclose(com_weighted, com_equal, equal_nan=True), \
        "신뢰도 컬럼이 없으면 결과가 같아야 함"
    print("✓ 테스트 통과 (신뢰도 없으면 fallback 동작 확인)")


def test_confidence_weighting_2d():
    """2D 신뢰도 가중치 테스트"""
    print("\n" + "="*60)
    print("Test 3: 2D 신뢰도 가중치")
    print("="*60)
    
    data = {
        'Nose_x': [100.0, 110.0],
        'Nose_y': [200.0, 210.0],
        'Nose_c': [0.85, 0.80],
        
        'LWrist_x': [150.0, 160.0],
        'LWrist_y': [250.0, 260.0],
        'LWrist_c': [0.92, 0.95],
        
        'RWrist_x': [160.0, 170.0],
        'RWrist_y': [260.0, 270.0],
        'RWrist_c': [0.45, 0.50],  # 낮은 신뢰도
    }
    
    df = pd.DataFrame(data)
    
    print("\n[신뢰도 가중치 사용]")
    com_weighted = compute_com_points_2d(df, use_confidence=True)
    
    print("\n[동일 가중치 사용]")
    com_equal = compute_com_points_2d(df, use_confidence=False)
    
    print(f"\nFrame 0 결과:")
    print(f"  신뢰도 가중치: {com_weighted[0]}")
    print(f"  동일 가중치:  {com_equal[0]}")
    print(f"  차이: {np.abs(com_weighted[0] - com_equal[0])}")
    
    assert not np.allclose(com_weighted, com_equal), "2D도 가중치가 다른 결과 생성해야 함"
    print("✓ 테스트 통과")


def test_ignore_joints():
    """특정 관절 무시하기 테스트"""
    print("\n" + "="*60)
    print("Test 4: 관절 무시 (ignore_joints)")
    print("="*60)
    
    data = {
        'Nose__x': [100.0],
        'Nose__y': [200.0],
        'Nose__z': [300.0],
        'Nose__c': [0.90],
        
        'LShoulder__x': [110.0],
        'LShoulder__y': [210.0],
        'LShoulder__z': [310.0],
        'LShoulder__c': [0.95],
    }
    
    df = pd.DataFrame(data)
    
    print("\n[모든 관절 포함]")
    com_all = compute_com_points_3d(df, ignore_joints=set(), use_confidence=True)
    
    print("\n[Nose 무시]")
    com_ignore = compute_com_points_3d(df, ignore_joints={'Nose'}, use_confidence=True)
    
    print(f"\n결과:")
    print(f"  모든 관절:     {com_all[0]}")
    print(f"  Nose 무시:    {com_ignore[0]}")
    
    assert not np.allclose(com_all, com_ignore), "무시된 관절로 결과 변경되어야 함"
    print("✓ 테스트 통과")


def test_nan_handling():
    """NaN 처리 테스트"""
    print("\n" + "="*60)
    print("Test 5: NaN 처리")
    print("="*60)
    
    data = {
        'LShoulder__x': [100.0, np.nan],
        'LShoulder__y': [200.0, np.nan],
        'LShoulder__z': [300.0, np.nan],
        'LShoulder__c': [0.95, np.nan],
        
        'RShoulder__x': [110.0, 120.0],
        'RShoulder__y': [210.0, 220.0],
        'RShoulder__z': [310.0, 320.0],
        'RShoulder__c': [0.90, 0.92],
    }
    
    df = pd.DataFrame(data)
    
    print("\n[NaN 포함 데이터]")
    com = compute_com_points_3d(df, use_confidence=True)
    
    print(f"\n결과:")
    print(f"  Frame 0 (LShoulder 유효): {com[0]}")
    print(f"  Frame 1 (LShoulder NaN): {com[1]}")
    
    # Frame 0은 유효한 값, Frame 1은 RShoulder만 사용
    assert not np.all(np.isnan(com[0])), "Frame 0은 유효한 값이어야 함"
    assert not np.all(np.isnan(com[1])), "Frame 1은 RShoulder만 사용해 유효해야 함"
    print("✓ 테스트 통과")


def main():
    print("\n" + "█"*60)
    print("COM 신뢰도 가중치 기능 테스트")
    print("█"*60)
    
    try:
        test_confidence_weighting_3d()
        test_confidence_missing_3d()
        test_confidence_weighting_2d()
        test_ignore_joints()
        test_nan_handling()
        
        print("\n" + "█"*60)
        print("✓ 모든 테스트 통과!")
        print("█"*60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
