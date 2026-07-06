# Grounding Evaluation 분석 보고서

## 📊 개요

Methods 텍스트에서 생성된 프로토콜 문장들의 근거를 찾는 평가입니다.
- **Grounded**: Methods에서 근거를 찾을 수 있는 문장
- **Hallucination**: Methods에서 근거를 찾을 수 없는 문장 (LLM이 만들어낸 내용)

---

## 🎯 핵심 결과 (Threshold 0.60)

| 모드 | Grounded | Hallucination | 평균 문장 수 | 평균 유사도 |
|------|----------|---------------|------------|---------|
| P1 (기본) | 64.3% | 35.7% | 19.4 | 0.607 |
| P2 (간단) | 41.5% | 58.5% | 12.2 | 0.490 |
| P3 | 58.3% | 41.7% | 16.2 | 0.587 |
| P4 | 56.3% | 43.7% | 16.8 | 0.583 |
| **P5** (다중회전) | **54.3%** | **45.7%** | 34.4 | 0.585 |
| **P6** (최적화) | **52.3%** | **47.7%** | 30.9 | 0.588 |

---

## 📈 상세 분석

### 1. Grounding Rate (Threshold별 비교)

```
Threshold 0.55 (가장 낮음 - 과도하게 관대):
  P1: 70.8%  P2: 47.0%  P3: 66.3%  P4: 65.3%  P5: 65.1%  P6: 67.1%

Threshold 0.60 (권장 - 적절한 엄격도):
  P1: 64.3%  P2: 41.5%  P3: 58.3%  P4: 56.3%  P5: 54.3%  P6: 52.3%

Threshold 0.65 (가장 높음 - 매우 엄격):
  P1: 54.0%  P2: 35.2%  P3: 47.1%  P4: 46.5%  P5: 40.6%  P6: 40.7%
```

### 2. 모드별 특징

#### P1 (기본 프롬프트)
- ✅ **최고 Grounding 율**: 64.3% (threshold 0.60)
- ✅ 가장 신뢰할 수 있는 생성
- ⚠️ 생성 문장 수 적음 (19.4개)
- 💡 **의미**: 기본 프롬프트가 Methods에 가장 충실하게 따름

#### P2 (간단한 버전)
- ❌ **가장 낮은 Grounding 율**: 41.5%
- ❌ **가장 높은 Hallucination 율**: 58.5%
- ⚠️ 생성 문장 수 최소 (12.2개)
- ⚠️ 평균 유사도 최저 (0.490)
- 💡 **의미**: 너무 간단한 프롬프트는 신뢰성 감소

#### P5 (다중 회전 CoV)
- 📊 평균적인 Grounding 율 (54.3%)
- ✅ 생성 문장 수 최다 (34.4개) → 더 상세함
- 😐 Hallucination 45.7% → 더 많은 비근거 콘텐츠
- 💡 **의미**: 더 자세하지만 환각이 증가

#### P6 (최적화)
- 📊 P5와 유사한 Grounding 율 (52.3%)
- 📊 생성 문장 수 중간 (30.9개)
- 😐 약간 높은 Hallucination (47.7%)
- 💡 **의미**: P5의 상세함을 유지하면서 품질 조정

---

## 🔍 해석

### Hallucination의 의미

**Hallucination (환각)이 높은 것 = 나쁜가?**

❌ 완전히 그렇지는 않음! 두 가지 가능성:

1. **실제 환각** (나쁜 것)
   - LLM이 Methods에 없는 내용을 만들어낸 경우
   - 예: "샘플을 liquid nitrogen에 보관" (Methods에 없음)

2. **의미적 추론** (좋을 수도 있음)
   - Methods의 내용을 기반으로 합리적인 추론
   - 예: "샘플을 냉동고에 보관" (Methods: "cool the sample" → 냉동은 자명한 추론)

### 왜 P1이 가장 "안전"한가?

P1 (기본 프롬프트)이 64.3%의 높은 Grounding 율을 가진 이유:

1. **보수적인 생성**: 더 짧고 직접적인 출력
2. **Methods에 충실**: 주어진 텍스트에 더 의존
3. **적은 창의성**: 추론이나 확장이 적음

→ **Trade-off**: Grounding ↑ but Coverage ↓

### 왜 P5/P6은 Hallucination이 높은가?

P5/P6 (다중 회전, CoV, ReAct)가 높은 Hallucination을 가진 이유:

1. **더 상세한 생성**: 더 많은 문장 (30-34개 vs 12-19개)
2. **적극적인 추론**: Methods에 암묵적인 내용도 추론
3. **전문가 스타일**: 실제 프로토콜처럼 보다 완전한 정보

→ **Trade-off**: Coverage ↑ (더 자세) but Grounding ↓

---

## 💡 권장사항

### 1. 사용 목적에 따라 선택

**높은 신뢰성 필요** (regulatory, compliance):
→ P1 사용 (64.3% Grounding)
- 장점: Methods에 명시된 것만
- 단점: 정보 부족

**완전한 프로토콜 필요** (실제 실험):
→ P5/P6 사용 (더 상세함)
- 장점: 더 실용적
- 단점: 일부 환각 (45-48%)

### 2. Hallucination 감소 방법

기존 코드 문제점 (항상 0이 나온 이유):
1. 생성된 프로토콜에서 문장을 제대로 추출하지 않음
2. 청킹 방식이 너무 엄격했음
3. 임계값이 너무 높았음 (보통 0.70+)

### 3. 개선 방향

✅ **이미 적용됨**:
- 문장 추출 개선 (ActionIR + protocol_text)
- 청킹 이중 전략 (문장 + 문자 기반)
- 적절한 임계값 (0.55, 0.60, 0.65)
- 상세 로깅 및 통계

⚠️ **추가 개선 가능**:
- Cross-Encoder 모델 사용 (더 정확한 유사도)
- Query expansion (문장을 여러 방식으로 표현)
- Domain-specific embedding 모델 (과학 논문 최적화)

---

## 📋 이전 스크립트의 문제점

### `eval_methods_grounding_protocols.py` (원본)의 문제:

```python
# ❌ 문제 1: 생성된 프로토콜에서 문장을 못 찾음
pred_sents = rec.get("sentences", []) or []  # 항상 빈 리스트!

# ❌ 문제 2: 청킹이 너무 큼 (1200자)
# 결과: 대부분의 짧은 문장이 매칭되지 않음

# ❌ 문제 3: 기본 임계값이 너무 높음 (0.60)
# 결과: Grounding rate가 매우 낮음

# 😕 결과: 모든 모드에서 0에 가까운 값
```

### `eval_methods_grounding_v2.py` (개선본)의 해결:

```python
# ✅ 해결 1: 다중 소스에서 문장 추출
# - ActionIR의 description
# - protocol_text 필드
# - 모든 문자열 필드를 재귀적으로 수집

# ✅ 해결 2: 이중 청킹 전략
# - 문장 기반: chunk_text_by_sentences()
# - 문자 기반: chunk_text_by_chars()
# - 결합: 두 방식의 장점 활용

# ✅ 해결 3: 적절한 임계값 비교
# - 0.55: 관대 (false positive 많음)
# - 0.60: 균형잡힘 (권장)
# - 0.65: 엄격 (false negative 많음)

# ✅ 결과: 의미있는 값들 (30-70%)
```

---

## 📊 CSV 파일 설명

### `per_protocol_grounding_v2.csv`
각 프로토콜별 상세 결과:
```
protocol_id, mode, n_sents, n_chunks, methods_length,
grounded_rate_0.55, hallucination_rate_0.55, grounded_0.55,
grounded_rate_0.60, hallucination_rate_0.60, grounded_0.60,
grounded_rate_0.65, hallucination_rate_0.65, grounded_0.65,
avg_max_sim_0.6, median_max_sim_0.6, ...
```

### `summary_modes_grounding_v2.csv`
모드별 평균 결과:
```
mode, n_sents (avg), n_chunks (avg),
grounded_rate_0.55 (avg), hallucination_rate_0.55 (avg),
...
```

---

## 🔚 결론

### 주요 발견사항

1. **P1 (기본)이 가장 신뢰할 수 있음**: 64.3% Grounding
2. **P2 (간단)은 피해야 함**: 58.5% Hallucination
3. **P5/P6은 상세하지만 환각 위험**: 45-48% Hallucination
4. **이전 평가 (모두 0%)는 잘못된 것**: 문장 추출 실패

### 권장 운영 방식

```
용도별 선택:
  ├─ Regulatory/Compliance → P1 (안전 우선)
  ├─ 학술 논문 → P3/P4 (균형)
  └─ 실제 실험용 → P5/P6 (완전성 우선)
      + Verifier로 환각 검증

Threshold 선택:
  ├─ 보수적: 0.65 (매우 엄격)
  ├─ 균형: 0.60 (권장)
  └─ 관대: 0.55 (빠른 필터링)
```

---

**생성일**: 2026-07-06  
**스크립트**: `eval_methods_grounding_v2.py`  
**모델**: `all-MiniLM-L6-v2`  
**Threshold**: 0.55, 0.60, 0.65
