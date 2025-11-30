# X-Factor 계산 수정 기록 (v4.0)

## 문제 보고
사용자가 제시한 JSON 응답에서 이상한 현상 발견:
- 초기 프레임 (0-6): 모두 90도로 clipping
- 스윙 중간: -50도 ~ +80도 (음수 값 포함)  
- 말기: 다시 90도로 clipping

**기대하는 동작**: 
- 초기/말기: 0도 근처 (중립자세는 상하체 회전 차이 거의 없음)
- 스윙 중간: 다양한 양수 값 (절댓값 기반)

## 근본 원인 분석

### 1단계: xfactor.py 검증 ✅
- `xfactor.py`의 `compute_xfactor()` 함수는 정상 작동
- 절댓값 적용 및 [0, 180] clipping 제대로 됨
- 테스트 결과: 초기 프레임 45.25° (이상 없음)

### 2단계: com_speed.py 발견 ❌
**문제 발견**: `com_speed.py`의 `compute_xfactor_by_planes()` 함수
```python
# OLD (v3.0 및 이전)
for name, (ax1, ax2) in planes.items():
    ang_sh = angles_deg_for_plane(sh, ax1, ax2)
    ang_pe = angles_deg_for_plane(pe, ax1, ax2)
    xf = ang_sh - ang_pe
    xf = smooth_median_then_moving(xf, w=5)
    out[name] = xf  # ← 절댓값 없음! 음수 그대로 반환
```

**문제의 결과**:
1. 각도 차이가 [-180, 180] 범위를 벗어나는 경우 처리 불완전
   - 예: 316.80° → -43.20° (정규화 후)
   - 그런데 스무딩으로 인한 보간이 극단값 생성
   - `np.clip(..., 0, 180)` 없음 → 90도로 clipping되는 듯한 현상

2. 음수 값이 JSON에 그대로 전달됨
   - `shift_frames[str(i)] = {"xfactor_deg": float(v_xf) ...}`
   - 여기서 `v_xf` = -48.5도 같은 음수값

### 3단계: 데이터 특성 파악
실제 CSV 분석 결과:
- Frame 0: shoulder angle = 173.69°, pelvis angle = -143.11°
- 각도 차: 173.69 - (-143.11) = 316.80°
- 정규화: 316.80 - 360 = -43.20° ✓ 맞음
- 절댓값: |-43.20| = **43.20°** ← 이게 진정한 X-Factor

**실제 현상**:
- 초기 자세에서 어깨가 이미 크게 회전되어 있음 (벗겨지는 포지션)
- 따라서 초기 X-Factor = 43-45° (0이 아님)
- 이것은 **데이터 특성** (자세 교정 필요), 코드 버그 아님

## v4.0 수정사항

### xfactor.py
1. **새로운 함수 추가** (라인 217-223):
```python
def normalize_angle_to_180(angle: float) -> float:
    """각도를 [-180, 180] 범위로 정규화 (여러 번 360 순환 처리)"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle
```

2. **compute_xfactor() 개선** (라인 283-293):
```python
# 4~6) 평면별 각도/언랩 → X-Factor
...
xf_raw = shoulder_angle - pelvis_angle
# [-180, 180] 범위 정규화 (여러 번 순환 처리)
xf_raw = np.array([normalize_angle_to_180(x) for x in xf_raw])
xf_smooth = smooth_median_then_moving(xf_raw, w=5)
# 절댓값 적용 (X-Factor는 항상 양수)
xf_smooth = np.abs(xf_smooth)
xf_smooth = np.clip(xf_smooth, 0, 180)
```

### com_speed.py
1. **normalize_angle_to_180() 함수 추가** (라인 184-189):
```python
def normalize_angle_to_180(angle: float) -> float:
    """각도를 [-180, 180] 범위로 정규화 (여러 번 360 순환 처리)"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle
```

2. **compute_xfactor_by_planes() 함수 재작성** (라인 191-219):
```python
def compute_xfactor_by_planes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """세 평면(X-Z, X-Y, Y-Z)에서 X-Factor 시퀀스 생성.
    ...
    v4.0 수정: 
    - 각도 차이를 [-180, 180] 범위로 정확히 정규화
    - 스무딩 후 절댓값 적용 (항상 양수 [0, 180])
    """
    ...
    for name, (ax1, ax2) in planes.items():
        ang_sh = angles_deg_for_plane(sh, ax1, ax2)
        ang_pe = angles_deg_for_plane(pe, ax1, ax2)
        xf = ang_sh - ang_pe
        # v4.0: [-180, 180] 범위로 정규화 (여러 번 순환 처리)
        xf = np.array([normalize_angle_to_180(x) for x in xf])
        xf = smooth_median_then_moving(xf, w=5)
        # v4.0: 절댓값 적용 (X-Factor는 항상 양수)
        xf = np.abs(xf)
        xf = np.clip(xf, 0, 180)
        out[name] = xf
```

## 테스트 결과

### 전에 (v3.0):
```
JSON Response (샘플):
Frame 0: "xfactor_deg": 90 (clipping artifact)
Frame 1: "xfactor_deg": 90
...
Frame 15: "xfactor_deg": -31.80 (음수!)
...
Frame 77: "xfactor_deg": -90 (음수 clipping)
```

### 후에 (v4.0):
```
Frame 0: 43.95° ✅ (양수, 초기 자세의 실제 값)
Frame 1: 44.07°
...
Frame 15: 32.89° ✅ (양수로 변환)
...
Frame 77: 90.00° ✅ (양수, 범위 내)

통계:
- Range: [0.91°, 145.06°] ✅
- Negative count: 0 ✅
- 90도 정확히: 1개만 (정상, 실제 값)
```

## 배포 절차

### 1. 캐시 삭제
```powershell
Remove-Item -Recurse -Force "d:\MyProjects\skeleton_metric-api\metric_algorithm\__pycache__"
```

### 2. Docker 재빌드
```bash
docker build -t skeleton-metric-api:v4.0 .
docker run -it skeleton-metric-api:v4.0
```

### 3. 검증
본 CSV 파일로 재계산하여 JSON 응답 확인:
- 모든 xfactor_deg > 0
- 범위 [0, 180]
- 초기 프레임 45° 근처
- 음수값 0개

## 핵심 교훈

1. **X-Factor 정의**: 항상 양수 (절댓값 기반)
2. **각도 정규화**: 여러 360° 순환이 필요할 수 있음
3. **스무딩 순서**: 절댓값 전에 스무딩해야 음수 생성 방지
4. **초기 자세**: 포스트 프로세싱이 정확하면 자세 교정은 coaching 문제

## 파일 변경사항

| 파일 | 라인 | 변경사항 | v이전 |
|------|------|--------|-------|
| xfactor.py | 217-223 | normalize_angle_to_180() 추가 | - |
| xfactor.py | 283-293 | compute_xfactor() 개선 | v3.0 |
| com_speed.py | 184-189 | normalize_angle_to_180() 추가 | - |
| com_speed.py | 191-219 | compute_xfactor_by_planes() 재작성 | v3.0 |

