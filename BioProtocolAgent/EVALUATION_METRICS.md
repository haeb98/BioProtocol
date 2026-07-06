# 📊 Evaluation Metrics 상세 설명서

## 📋 목차
1. [Prompt Variants (P1-P6)](#-prompt-variants-p1-p6)
2. [Action Extraction Metrics](#-action-extraction-metrics)
3. [Grounding & Hallucination Metrics](#-grounding--hallucination-metrics)
4. [Natural Language Protocol Metrics](#-natural-language-protocol-metrics)
5. [전체 평가 파이프라인](#-전체-평가-파이프라인)

---

## 🔤 Prompt Variants (P1-P6)

### 개요

각 변이체는 Methods 텍스트에서 프로토콜을 생성하는 서로 다른 **파이프라인 구성**을 나타냅니다.

```
Methods Text (input)
    ↓
[P1-P6 중 하나]
    ↓
Protocol + ActionIR (output)
```

### P1: Direct Writing (기본 프롬프트)

**구성**:
```
Methods → [Writer (Simple)]
```

**특징**:
- ✅ 가장 간단한 파이프라인
- ✅ 단일 LLM 호출
- ✅ Methods를 읽고 직접 자연어 프로토콜 생성
- ⚠️ 구조화 정보 없음
- ⚠️ 액션 분해 없음

**프롬프트**:
```python
"논문의 Methods 섹션을 읽고 단계별 실험 프로토콜을 생성하세요.
각 단계는 다음을 포함해야 합니다:
- 수행할 동작 (Add, Incubate, Mix 등)
- 사용할 재료
- 온도, 시간 등 조건
- 예상 결과"
```

**결과 특성**:
- 평균 문장 수: 19.4개
- Grounding Rate: 64.3% (가장 높음)
- Hallucination Rate: 35.7% (가장 낮음)
- 평균 유사도: 0.607

**언제 사용?**:
- ✅ 높은 신뢰성 필요 (regulatory, compliance)
- ✅ Methods에 명시된 것만 필요
- ❌ 완전한 프로토콜 필요할 때는 부족

---

### P2: Prompted Writing (간단한 버전)

**구성**:
```
Methods → [Writer (Prompted)]
```

**특징**:
- ⚠️ 더 간단한 프롬프트
- ⚠️ 최소 정보만 추출
- ❌ 신뢰성 매우 낮음
- ❌ 많은 정보 누락

**프롬프트**:
```python
"Methods에서 실험 단계들을 찾아 간단히 나열하세요.
각 단계는 한 문장으로."
```

**결과 특성**:
- 평균 문장 수: 12.2개 (가장 적음)
- Grounding Rate: 41.5% (가장 낮음)
- Hallucination Rate: 58.5% (가장 높음) ❌
- 평균 유사도: 0.490 (가장 낮음)

**언제 사용?**:
- ❌ 거의 사용하지 않음
- 🔬 비교 대상일 뿐 (얼마나 나쁜지 보기 위해)

---

### P3: Task Mining + Skeleton Writer

**구성**:
```
Methods → [Task Planner] → [Writer (Skeleton)]
       (태스크 추출)   (구조화된 생성)
```

**특징**:
- ✅ 2단계 파이프라인
- ✅ Task 추출으로 구조화
- ✅ 기본 Task 기반 생성
- ⚠️ Step이나 Action 레벨 정보 없음

**동작**:
1. Task Planner: Methods에서 주요 작업 추출 (3-10개)
   ```
   예: "Sample preparation", "PCR amplification", "Gel electrophoresis"
   ```

2. Writer (Skeleton): Task 기반 프로토콜 생성
   ```
   각 Task마다 하위 스텝들을 자동 생성
   ```

**결과 특성**:
- 평균 문장 수: 16.2개
- Grounding Rate: 58.3%
- Hallucination Rate: 41.7%
- 평균 유사도: 0.587

**언제 사용?**:
- ✅ 기본 구조화 필요할 때
- ✅ Task 수준의 개요 제공

---

### P4: Task + Step Planning

**구성**:
```
Methods → [Task Planner] → [Step Planner] → [Writer (Skeleton)]
       (태스크)        (세부 단계)    (생성)
```

**특징**:
- ✅ 3단계 파이프라인
- ✅ 계층적 구조 (Task → Step)
- ✅ 각 Task 내 세부 단계 구성
- ⚠️ 원자적 Action 수준 아직 없음

**동작**:
1. Task Planner: 주요 작업 추출
2. Step Planner: 각 Task를 세부 단계로 분해
   ```
   Task: "Sample preparation"
   ├─ Step 1: "Add sample to tube"
   ├─ Step 2: "Mix well"
   └─ Step 3: "Incubate at room temperature"
   ```
3. Writer: Step 기반 프로토콜 생성

**결과 특성**:
- 평균 문장 수: 16.8개
- Grounding Rate: 56.3%
- Hallucination Rate: 43.7%
- 평균 유사도: 0.583

**언제 사용?**:
- ✅ 계층적 구조 필요할 때
- ✅ Task와 Step 수준의 명확성 필요

---

### P5: Full Pipeline + Action Extraction

**구성**:
```
Methods → [Task Planner] → [Step Planner] → [Action Extractor] → [Writer]
       (태스크)        (단계)         (액션 분해)        (생성)
```

**특징**:
- ✅ 4단계 파이프라인 (전체 계층 구조)
- ✅ 원자적 Action 수준까지 분해
- ✅ 각 Action마다 재료, 조건, 산물 추출
- ✅ 가장 상세한 구조화
- ⚠️ 검증 없음 (오류 검사 미흡)
- ⚠️ Hallucination 증가 가능성

**동작**:
1. Task Planner: 주요 작업 추출
2. Step Planner: 각 Task를 세부 단계로 분해
3. Action Extractor: 각 Step을 원자적 액션으로 분해
   ```
   Step: "Mix well"
   ├─ Action: "Add"
   │  └─ Materials: [tube, sample]
   │  └─ Conditions: []
   └─ Action: "Mix"
      └─ Conditions: [duration: 5 min, speed: high]
   ```
4. Writer: 액션 기반 자연어 프로토콜 생성

**ActionIR 예시**:
```json
{
  "action_id": "Bio-protocol-2302::T1::S1::A1",
  "action": "Add",
  "description": "Add RNA sample to lysis buffer",
  "materials": [
    {"name": "RNA sample", "role": "substrate", "volume": "100 μL"},
    {"name": "lysis buffer", "role": "reagent"}
  ],
  "conditions": [
    {"type": "temperature", "value": "room temperature"}
  ],
  "produces": ["lysed RNA sample"],
  "evidence_span": "Add the RNA sample to lysis buffer..."
}
```

**결과 특성**:
- 평균 문장 수: 34.4개 (가장 많음)
- Grounding Rate: 54.3%
- Hallucination Rate: 45.7%
- 평균 유사도: 0.585
- **Step F1**: 0.816 (가장 높음!) ✅

**언제 사용?**:
- ✅ 완전한 액션 기반 프로토콜 필요
- ✅ ActionIR이 필요한 응용 분야
- ✅ 세부 정보가 중요한 경우
- ⚠️ Hallucination 위험이 있으므로 검증 필요

---

### P6: Full Pipeline + Verification (최적화 버전)

**구성**:
```
Methods → [Task] → [Step] → [Action] → [Verifier (CoV)] → [Writer]
                                      (검증 단계 추가!)
```

**특징**:
- ✅ P5에 검증 단계 추가
- ✅ Chain-of-Verification (CoV)으로 생성된 액션 검증
- ✅ 환각 탐지 및 수정
- ✅ P5보다 품질 개선 (일부 비근거 액션 제거)
- ⚠️ 추가 LLM 호출로 비용 증가

**검증 프로세스**:
```
각 Action에 대해:
1. Verification questions 생성
   "이 액션이 Methods에서 지원되는가?"
   
2. Evidence 찾기
   Methods에서 관련 문구 검색
   
3. Verdict 결정
   - supported: Methods에 명시됨
   - partially_supported: 관련 내용 있음
   - hallucinated: 근거 없음
   - contradicted: 다른 내용 있음
   
4. 수정
   hallucinated/contradicted는 수정 또는 제거
```

**검증 결과 예시**:
```json
{
  "action": {...},
  "verification": {
    "global_verdict": "supported",
    "verification_questions": [
      "이 온도가 Methods에 명시되어 있는가?",
      "이 시간이 맞는가?"
    ],
    "reasoning_traces": [
      {
        "question": "이 온도가 Methods에 명시되어 있는가?",
        "thought": "37°C incubation을 찾음",
        "evidence_spans": ["Incubate at 37°C for 2 hours"],
        "local_verdict": "supported"
      }
    ],
    "revision_suggestion": ""
  }
}
```

**결과 특성**:
- 평균 문장 수: 30.9개
- Grounding Rate: 52.3%
- Hallucination Rate: 47.7%
- 평균 유사도: 0.588
- **Step F1**: 0.741

**언제 사용?**:
- ✅ 최고의 품질 필요할 때 (학술 논문, 규제)
- ✅ 환각 탐지가 중요할 때
- ✅ Methods 근거 추적 필수
- ❌ 비용이 높음 (추가 검증 LLM 호출)

---

### P1-P6 비교 표

| 특성 | P1 | P2 | P3 | P4 | P5 | P6 |
|------|----|----|----|----|----|----|
| **파이프라인 단계** | 1 | 1 | 2 | 3 | 4 | 5 |
| **포함 노드** | Writer | Writer | Task+Writer | Task+Step+Writer | Task+Step+Action+Writer | Task+Step+Action+Verifier+Writer |
| **구조화 수준** | 없음 | 없음 | Task | Task+Step | Task+Step+Action | Task+Step+Action+Verification |
| **ActionIR** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **검증** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (CoV) |
| **평균 문장 수** | 19.4 | 12.2 | 16.2 | 16.8 | 34.4 | 30.9 |
| **Grounding Rate** | 64.3% | 41.5% | 58.3% | 56.3% | 54.3% | 52.3% |
| **Hallucination Rate** | 35.7% | 58.5% | 41.7% | 43.7% | 45.7% | 47.7% |
| **Step F1** | - | - | - | - | 0.816 | 0.741 |
| **추천 사용** | 신뢰성↑ | ❌ | 기본 구조 | 계층 구조 | 완전한 액션 | 최고 품질 |

---

## 📊 Action Extraction Metrics

### 1. Step-Level Metrics

#### Step Precision
```
정의: 올바르게 감지된 Step의 비율
계산: TP / (TP + FP)
범위: 0.0 - 1.0

예:
  Gold Steps: [S1, S2, S3]
  Predicted: [S1, S2, S3, S4_wrong]
  
  TP = 3 (S1, S2, S3 맞게 찾음)
  FP = 1 (S4는 틀림)
  
  Precision = 3 / (3 + 1) = 0.75
```

#### Step Recall
```
정의: 찾아야 할 Step 중 실제로 찾은 비율
계산: TP / (TP + FN)
범위: 0.0 - 1.0

예:
  Gold Steps: [S1, S2, S3]
  Predicted: [S1, S2]
  
  TP = 2 (S1, S2 맞게 찾음)
  FN = 1 (S3를 못 찾음)
  
  Recall = 2 / (2 + 1) = 0.67
```

#### Step F1
```
정의: Precision과 Recall의 조화평균
계산: 2 * (Precision * Recall) / (Precision + Recall)
범위: 0.0 - 1.0

P5 결과: 0.816 (매우 우수!)
P6 결과: 0.741 (우수)

해석:
  - F1 > 0.8: 매우 우수
  - F1 > 0.7: 우수
  - F1 > 0.5: 양호
  - F1 < 0.5: 부족
```

**구현**:
```python
# 문장 기반 매칭 (Sentence-Transformer)
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# 임베딩 계산
gold_emb = model.encode(gold_steps)
pred_emb = model.encode(predicted_steps)

# 유사도 행렬 계산
similarity = util.cos_sim(pred_emb, gold_emb)

# 최대 유사도를 기준으로 매칭 (threshold = 0.7)
matches = (similarity.max(dim=1).values > 0.7).sum()
```

### 2. Material Metrics

#### Material Soft IOU
```
정의: 예측된 재료와 Gold 재료의 의미적 교집합
범위: 0.0 - 1.0

계산 과정:
1. 각 예측 재료의 임베딩 계산
2. 각 Gold 재료와의 최대 유사도 찾기
3. 유사도 임계값 (0.75) 이상인 것들만 매칭
4. Soft IOU = 매칭된 수 / 전체 재료 수

예:
  Gold: [RNA, buffer, water]
  Pred: [RNA sample, lysis buffer, NaCl] (추가됨)
  
  매칭:
    - RNA sample ↔ RNA (유사도: 0.88) ✓
    - lysis buffer ↔ buffer (유사도: 0.92) ✓
    - NaCl ↔ water (유사도: 0.45) ✗
  
  Soft IOU = 2 / 3 = 0.667
```

P5 결과: 0.415 (양호)
P6 결과: 0.456 (양호)

### 3. Condition Metrics

#### Condition Soft IOU
```
정의: 예측된 조건과 Gold 조건의 의미적 교집합
범위: 0.0 - 1.0

조건 유형:
  - temperature: "37°C", "room temperature"
  - duration: "2 hours", "overnight"
  - speed: "high speed", "3000 rpm"
  - pH: "pH 7.0"

계산: Material과 동일한 방식
  - 각 조건의 임베딩 계산
  - 유사도 기반 매칭
  - Soft IOU 계산
```

P5 결과: 0.531 (양호)
P6 결과: 0.532 (양호)

---

## 📊 Grounding & Hallucination Metrics

### 1. Grounded Rate

```
정의: Methods에서 근거를 찾을 수 있는 문장의 비율
계산: grounded_sentences / total_sentences
범위: 0.0 - 1.0 (0% - 100%)

예:
  생성된 문장 5개:
  1. "Add sample to tube" → Methods에 "Add sample" 있음 ✓ grounded
  2. "Mix at 37°C" → Methods에 "37°C" 있음 ✓ grounded
  3. "Spin at 5000 rpm" → Methods에 없음 ❌ hallucinated
  4. "Cool to 4°C" → Methods에 "cool" 있음 ✓ grounded
  5. "Use liquid nitrogen" → Methods에 없음 ❌ hallucinated
  
  Grounded Rate = 3 / 5 = 60%
```

**계산 방식**:

```python
# 1. Methods를 청크로 분할
chunks = chunk_text_by_sentences(methods_text)

# 2. 각 문장과 청크의 유사도 계산
sent_emb = model.encode(sentences)
chunk_emb = model.encode(chunks)
similarity = util.cos_sim(sent_emb, chunk_emb)

# 3. 최대 유사도가 임계값 이상이면 grounded
threshold = 0.60
grounded_count = (similarity.max(axis=1) >= threshold).sum()

grounded_rate = grounded_count / len(sentences)
```

### 2. Hallucination Rate

```
정의: Methods에서 근거를 찾을 수 없는 문장의 비율
계산: 1 - Grounded Rate
범위: 0.0 - 1.0 (0% - 100%)

Hallucination Rate = hallucinated_sentences / total_sentences

예: Grounded Rate가 60%이면
    Hallucination Rate = 1 - 0.60 = 0.40 (40%)
```

**두 가지 유형**:

1. **진정한 환각** (나쁜 것)
   - "liquid nitrogen에 냉동" (Methods에 없음)
   - LLM이 만든 잘못된 정보

2. **의미적 추론** (좋을 수도)
   - "냉동 보관" ← Methods: "cool the sample"
   - 합리적인 전문가 추론
   - 생략된 암묵적 지식

### 3. Threshold Comparison

3가지 임계값으로 엄격도 비교:

```
Threshold 0.55 (관대):
  - False Positive (환각이라고 봐야 할 것을 grounded로 봄) 많음
  - Grounding Rate 높음 (과도하게 높음)
  - 사용: 빠른 필터링

Threshold 0.60 (권장):
  - 정확도와 재현율의 균형
  - 현실적인 환각 탐지
  - 사용: 실제 평가

Threshold 0.65 (엄격):
  - False Negative (근거가 있는데 hallucinated로 봄) 많음
  - 매우 엄격한 판단
  - 사용: 매우 신뢰할 수 있는 내용만
```

P1 결과 (Threshold 0.60):
```
Threshold 0.55: Grounded = 70.8%, Hallucination = 29.2%
Threshold 0.60: Grounded = 64.3%, Hallucination = 35.7%
Threshold 0.65: Grounded = 54.0%, Hallucination = 46.0%
```

### 4. Similarity Statistics

```
Average Max Similarity (평균 최대 유사도):
  각 문장과 Methods 청크의 최대 유사도의 평균
  범위: 0.0 - 1.0
  P1: 0.607 (높음 = Methods와 유사함)
  P2: 0.490 (낮음 = Methods와 거리 있음)

Median Max Similarity (중간값 최대 유사도):
  최대 유사도들의 중간값
  분포를 나타냄
  P1: 0.645 (대부분의 문장이 Methods와 유사)

Min/Max Similarity (범위):
  최소 유사도와 최대 유사도
  다양성 측정
```

---

## 📊 Natural Language Protocol Metrics

### 1. BLEU Score

```
정의: 생성된 프로토콜과 Gold 프로토콜 간의 n-gram 일치
범위: 0.0 - 1.0 (0 - 100)

계산:
  1. 1-gram 일치도 계산
  2. 2-gram 일치도 계산
  3. 3-gram 일치도 계산
  4. 4-gram 일치도 계산
  5. 가중 평균 (일반적으로 균등 가중)

예:
  Gold: "Add the sample to the tube and mix well"
  Generated: "Add sample to tube and mix"
  
  1-gram: 7/8 = 0.875 (맞는 단어들)
  2-gram: 5/7 = 0.714 ("Add sample", "to tube" 등)
  
  BLEU ≈ (0.875 + 0.714 + ...) / 4

주의: BLEU는 다양성이 낮은 방식
      같은 의미라도 다르게 쓰면 낮은 점수
```

**해석**:
- BLEU > 0.5: 우수
- BLEU > 0.3: 양호
- BLEU < 0.3: 부족

### 2. ROUGE Score

```
정의: 생성된 프로토콜과 Gold 프로토콜 간의 회수율
범위: 0.0 - 1.0

ROUGE-L (Longest Common Subsequence):
  가장 긴 공통 부분 수열을 이용
  순서는 중요하지만 연속성은 필요 없음
  
  예:
    Gold: "Add sample to tube and incubate at 37°C"
    Generated: "Add sample to tube and heat to 37°C"
    
    공통: "Add sample to tube and ... 37°C"
    ROUGE-L = 공통 길이 / 전체 길이

ROUGE-N:
  N-gram 기반 회수율
  ROUGE-1: 단어 단위
  ROUGE-2: 인접한 두 단어
```

**해석**:
- 더 의미론적 (BLEU보다)
- 다양한 표현 방식을 인정

### 3. Step-Level F1

```
정의: 생성된 프로토콜의 Step들과 Gold의 Step들 간의 F1
범위: 0.0 - 1.0

계산:
1. 생성된 프로토콜을 문장/단계로 분할
2. Gold 프로토콜도 분할
3. 각 단계의 임베딩 계산
4. Sentence Similarity (Threshold 0.7) 기반 매칭
5. Precision, Recall, F1 계산

예:
  Gold Steps:
    S1: "Add RNA to buffer"
    S2: "Incubate at 37°C"
    S3: "Cool to room temperature"
  
  Generated Steps:
    S1: "Add RNA sample to lysis buffer"
    S2: "Incubate at 37°C for 2 hours"
    S3: "Cool down to room temperature"
    S4: "Spin at 5000 rpm" (extra)
  
  매칭 (유사도 > 0.7):
    Generated S1 ↔ Gold S1 ✓ (유사도: 0.85)
    Generated S2 ↔ Gold S2 ✓ (유사도: 0.82)
    Generated S3 ↔ Gold S3 ✓ (유사도: 0.78)
    Generated S4: 매칭 없음
  
  TP = 3, FP = 1, FN = 0
  Precision = 3/(3+1) = 0.75
  Recall = 3/(3+0) = 1.0
  F1 = 2*(0.75*1.0)/(0.75+1.0) = 0.857
```

---

## 🔄 전체 평가 파이프라인

### 평가 흐름

```
생성된 프로토콜 (P1-P6)
    ↓
┌─────────────────────────────┐
│ 1. Action Extraction 평가     │
├─────────────────────────────┤
│ - Step Precision/Recall/F1  │
│ - Material IOU              │
│ - Condition IOU             │
│ 스크립트: eval_ablation_actions_v2.py
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ 2. 자연어 프로토콜 평가       │
├─────────────────────────────┤
│ - BLEU Score                │
│ - ROUGE Score               │
│ - Step-level F1             │
│ 스크립트: eval_hier_protocol_vs_generated.py
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ 3. Grounding & Hallucination│
├─────────────────────────────┤
│ - Grounded Rate             │
│ - Hallucination Rate        │
│ - Similarity Statistics     │
│ 스크립트: eval_methods_grounding_v2.py
└─────────────────────────────┘
    ↓
최종 평가 결과 (CSV + 분석)
```

### 실행 명령

```bash
# 1. Action Extraction 평가
python scripts/eval_ablation_actions_v2.py
# 출력: data/ablation_eval_actions_v2.csv

# 2. 자연어 프로토콜 평가
python scripts/eval_hier_protocol_vs_generated.py
# 출력: reports/protocol_eval/

# 3. Grounding 평가
python scripts/eval_methods_grounding_v2.py
# 출력: reports/grounding_eval/per_protocol_grounding_v2.csv
#       reports/grounding_eval/summary_modes_grounding_v2.csv
```

---

## 📋 CSV 파일 구조

### ablation_eval_actions_v2.csv

```
protocol_id, n_gold, n_pred, 
step_precision, step_recall, step_f1,
order_score,
mat_iou, cond_iou,
grounding_hallucination_rate, evidence_coverage,
mode, has_methods_text

예:
Bio-protocol-2302, 24, 15, 
1.0, 0.625, 0.769,
0.581,
0.467, 0.467,
0.5, 0.0,
P1, True
```

**컬럼 설명**:
- `n_gold`: Gold 액션 수
- `n_pred`: 생성된 액션 수
- `step_f1`: 단계 추출 정확도 (0-1)
- `order_score`: 액션 순서 일관성 (0-1)
- `mat_iou`: 재료 매칭 점수 (0-1)
- `cond_iou`: 조건 매칭 점수 (0-1)
- `mode`: P1-P6

### per_protocol_grounding_v2.csv

```
mode, protocol_id, n_sents, n_chunks, methods_length,
grounded_rate_0.55, hallucination_rate_0.55, grounded_0.55,
grounded_rate_0.6, hallucination_rate_0.6, grounded_0.6,
grounded_rate_0.65, hallucination_rate_0.65, grounded_0.65,
avg_max_sim_0.6, median_max_sim_0.6, ...

예:
P1, Bio-protocol-2302, 19, 29, 6042,
0.684, 0.316, 13,
0.474, 0.526, 9,
0.421, 0.579, 8,
0.608, 0.589, ...
```

---

## 🎯 해석 가이드

### P1-P6 선택 기준

**신뢰성 우선** (높은 Grounding) → **P1**
```
✅ 장점:
  - 가장 신뢰할 수 있음 (64.3% grounded)
  - 환각 적음 (35.7%)
  - Methods에 충실

❌ 단점:
  - 정보 부족할 수 있음 (19.4개 문장만)
  - 불완전한 프로토콜
```

**완전성 우선** (더 자세한) → **P5 또는 P6**
```
✅ 장점:
  - 더 상세함 (30-34개 문장)
  - 높은 Step F1 (0.74-0.82)
  - 실제 프로토콜에 가까움

❌ 단점:
  - 환각 위험 (45-48%)
  - P6은 비용 높음 (검증 LLM 호출)
```

**규제 준수** (매우 엄격) → **P1 + P6 조합**
```
1. P1으로 기본 프로토콜 생성
2. P6으로 검증해서 추가 정보 보충
3. 검증된 것만 포함
```

---

**생성일**: 2026-07-06  
**버전**: 1.0  
**최종 업데이트**: Grounding v2 평가 추가
