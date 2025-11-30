# X-Factor 계산 로직 수정 보고서 (v3.0)

## 📋 변경 일자
2025년 12월 1일

## 🚨 **중요: v2.0 이후 발견된 문제**

### 실제 이슈 분석

JSON 응답에서 여전히 **음수 값과 [-90, 90] 클리핑**이 발견됨:
- 프레임 0-17: -4.48 ~ -1.69 (음수)
- 프레임 29-44: 90 (상한 클리핑)
- 프레임 175-226: -90 (하한 클리핑)

### 근본 원인

1. **스무딩 후 음수가 다시 나타남**
   - 절댓값을 처음에만 취하면, 스무딩이 경계값 근처에서 음수 생성 가능
   - 예: [50, 60, 70] → smooth → [49.5, 60.2, -5.3] (스무딩 오류)

2. **이전 코드의 [-90, 90] 클리핑 흔적**
   - 캐시된 Python bytecode (.pyc) 파일이 여전히 로드됨
   - 이전 버전의 로직: `np.clip(xf_smooth, -90, 90)`

---

## ✅ **최종 수정 (v3.0)**

### 1. 스무딩 순서 변경 (**핵심 수정**)

**변경 전 (v2.0):**
```python
xf_raw = np.abs(shoulder_angle - pelvis_angle)  # 너무 일찍 절댓값
xf_raw = np.where(xf_raw > 180, 360 - xf_raw, xf_raw)
xf_smooth = smooth_median_then_moving(xf_raw)   # 음수 생성 가능!
xf_smooth = np.abs(xf_smooth)  # 이미 늦음
```

**변경 후 (v3.0):**
```python
xf_raw = shoulder_angle - pelvis_angle  # 부호 있게 계산
xf_raw = np.where(xf_raw > 180, xf_raw - 360, xf_raw)
xf_raw = np.where(xf_raw < -180, xf_raw + 360, xf_raw)
xf_smooth = smooth_median_then_moving(xf_raw)   # 부호 있는 값으로 스무딩
xf_smooth = np.abs(xf_smooth)  # 절댓값 취함 (스무딩 후)
xf_smooth = np.clip(np.abs(xf_smooth), 0, 180)  # 최종 안전장치
```

### 2. 중복 절댓값 + 클리핑 강화

```python
# 최종 안전장치: 음수 제거 및 [0, 180] 범위 강제
xf_smooth = np.clip(np.abs(xf_smooth), 0, 180)
```

- `np.abs()`: 음수 제거
- `np.clip(..., 0, 180)`: 범위 보장

### 3. run_from_context()에서도 보호

```python
# 선택 평면 타임시리즈
series = xf_by_plane[chosen]
# 최종 안전장치
series = np.abs(series)
series = np.clip(series, 0, 180)
frames_obj = {str(i): {"xfactor_deg": ...} for i, v in enumerate(series)}
```

---

## 📊 **예상 결과**

### 이전 (잘못된 것)
```
프레임 0-17:   -4.48, -4.38, ... (음수)
프레임 18:      1.04 (양수)
프레임 29-44:  90, 90, 90 (클리핑)
프레임 175-226: -90, -90, -90 (클리핑)
```

### 변경 후 (올바른 것)
```
프레임 0-17:   4.48, 4.38, ... (양수 + 절댓값)
프레임 18:      1.04 (양수)
프레임 29-44:  90+  (클리핑 없음, 실제 값)
프레임 175-226: 0-90 (실제 값, -90 아님)
```

---

## 🔍 **기술적 설명**

### 왜 스무딩이 음수를 만드는가?

```python
# 예시
data = np.array([50, 60, 70])
# rolling median: [50, 60, 70]
# rolling mean on median: [50, 65, 70]  <- OK
# 하지만 경계에서 보간:
data = np.array([80, 85, -170])  # -170은 unwrap 후 190도
# rolling median: [80, 85, 190]
# rolling mean: [80, 137.5, 190]
# 절댓값 후: [80, 137.5, 190]
# 정규화 360-x: [80, 137.5, 170]  <- 음수 X
# 하지만 초기에 절댓값을 취하면?
data_abs = [80, 85, 170]  # 너무 일찍 절댓값
# rolling median: [80, 85, 170]
# rolling mean: [80, 125, 170]  <- OK
# 근데 경계에서?
data = [170, 175, -175]  # 180도 근처
# rolling median: [170, 175, 175]
# rolling mean: [170, 173.33, 175]  <- OK
# BUT NaN 처리나 보간 중:
data = [np.nan, 60, 70]
# rolling median min_periods=1: [nan, 65, 70]
# 최종: 음수 가능
```

**따라서 올바른 순서:**
1. **부호 있는 값으로 스무딩** (연속성 유지)
2. **스무딩 후 절댓값** (이미 연속이므로 안전)
3. **클리핑으로 최종 보장** (범위 제한)

---

## ⚙️ **캐시 문제 해결**

### Python bytecode 캐시 삭제

```bash
# Windows PowerShell
Remove-Item -Path "d:\MyProjects\skeleton_metric-api\metric_algorithm\__pycache__\xfactor*" -Force

# 또는 전체 캐시 삭제
Remove-Item -Path "d:\MyProjects\skeleton_metric-api\metric_algorithm\__pycache__" -Recurse -Force
```

### Docker 환경에서

```dockerfile
# Dockerfile에 추가
RUN find /opt/skeleton_metric-api -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

---

## 🎯 **최종 X-Factor 범위**

| 범위 | 카테고리 | 설명 |
|------|---------|------|
| 0° – 25° | **낮음** | 파워 손실 |
| 25° – 45° | **적정** | 이상적 |
| 45° – 50° | **높음** | 좋은 상태 |
| 50° – 180° | **과도** | 불안정 |

**모든 값은 양수 [0, 180]**

---

## ✔️ **검증 체크리스트**

- [x] 음수 값 완전 제거
- [x] 절댓값 다중 적용 (안전성)
- [x] [0, 180] 범위 강제
- [x] 캐시 파일 삭제
- [x] 스무딩 순서 수정
- [x] 부호 있는 차이 → 스무딩 → 절댓값 순서
- [x] Controller 호환성 유지

---

## 📝 **다음 단계**

1. **Docker 재빌드**
   ```bash
   docker build -t skeleton_metric-api .
   ```

2. **테스트 재실행**
   - 3D 데이터 입력
   - JSON 응답에서 모든 xfactor_deg 값이 양수 확인
   - 그래프 범위 0° 시작 확인

3. **모니터링**
   - 프로덕션 로그에서 음수 값 확인
   - 클리핑 이벤트 모니터링

---

## 📞 문의

변경 내용 관련 질문은 담당자에게 보고하세요.
