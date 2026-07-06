# BioProtocolAgent: 멀티에이전트 기반 바이오 프로토콜 자동 생성

> 과학 논문의 Methods 섹션에서 자동으로 **재현 가능한 바이오 실험 프로토콜**을 생성하는 AI 시스템

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Submitted-brightgreen)]()

---

## 🎯 프로젝트 개요

**BioProtocolAgent**는 **LangGraph 기반의 멀티에이전트 프레임워크**로, 생물학 논문의 Methods 텍스트를 구조화된 바이오 프로토콜로 변환합니다.

### 핵심 기능
- ✅ **Task Mining**: 논문에서 핵심 실험 작업 자동 추출
- ✅ **Step Planning**: 작업을 단계별 절차로 구조화
- ✅ **Action Extraction**: 각 단계를 원자적 액션(Add, Incubate 등)으로 분해
- ✅ **Grounding Verification**: 생성된 내용이 Methods 텍스트에 근거를 가지는지 검증
- ✅ **Hallucination Detection**: LLM 환각 탐지 및 정량 평가
- ✅ **Protocol Generation**: 최종 자연어 프로토콜 생성

---

## 🏗️ 시스템 아키텍처

### 에이전트 워크플로우

```
Methods Text (논문)
    ↓
[Task Planner] → 핵심 태스크 추출
    ↓
[Step Planner] → 단계별 구조화
    ↓
[Action Extractor] → 원자적 액션 분해
    ↓
[Condition Extractor] → 온도, 시간 등 조건 추출
    ↓
[Verifier (CoV)] → Chain-of-Verification으로 검증
    ↓
[Writer] → 자연어 프로토콜 생성
    ↓
Structured Action IR + Natural Language Protocol
```

### 노드 설명

| 노드 | 역할 | LLM | 입력 | 출력 |
|------|------|-----|------|------|
| **Task Planner** | Methods에서 핵심 Task 추출 | GPT-4-1106 | Methods, 제목 | Task 리스트 |
| **Step Planner** | Task를 세부 Step으로 구조화 | GPT-4o-mini | Task + Methods | StepIR |
| **Action Extractor** | Step을 원자적 Action으로 분해 | GPT-4o-mini | Step + Methods | ActionIR |
| **Condition Extractor** | 물리적 조건(온도, 시간 등) 추출 | GPT-4o-mini | Action | ConditionIR |
| **Verifier** | 생성된 내용을 Methods 텍스트로 검증 (CoV+ReAct) | GPT-4o-mini | Action + Methods | Verification 결과 |
| **Writer** | ActionIR을 자연어 프로토콜로 변환 | GPT-4o-mini | ActionIR | Natural protocol |

---

## 📊 성능 결과

### 평가 메트릭

#### 1. Action 추출 정확도 (Ablation Study)
- **Step F1**: 단계 추출 정확도 (0-1)
- **Material IOU**: 재료 매칭 (Soft Intersection-over-Union)
- **Condition IOU**: 조건 매칭 (온도, 시간 등)

#### 2. Grounding & Hallucination (개선된 평가)
- **Grounded Rate**: Methods에서 근거를 찾을 수 있는 문장 비율
- **Hallucination Rate**: Methods에서 근거를 찾을 수 없는 문장 비율
- **Average Max Similarity**: 평균 최대 유사도 (0-1)

### 최고 성능 결과

#### Action Extraction (Ablation Study)

| 메트릭 | P1 (기본) | P5 (다중회전) | P6 (최적) |
|--------|----------|--------------|----------|
| **Step F1** | 0.494 | **0.816** | **0.741** |
| **Material IOU** | 0.247 | **0.415** | **0.456** |
| **Condition IOU** | 0.573 | 0.531 | 0.532 |

#### Grounding Evaluation (Threshold 0.60)

| 모드 | Grounded Rate | Hallucination Rate | 평균 문장 수 | 평균 유사도 |
|------|--------|---------|---------|---------|
| **P1** (기본) | **64.3%** | 35.7% | 19.4 | 0.607 |
| P2 (간단) | 41.5% | **58.5%** | 12.2 | 0.490 |
| P3 | 58.3% | 41.7% | 16.2 | 0.587 |
| P4 | 56.3% | 43.7% | 16.8 | 0.583 |
| P5 (다중회전) | 54.3% | 45.7% | 34.4 | 0.585 |
| P6 (최적화) | 52.3% | 47.7% | 30.9 | 0.588 |

**핵심 발견사항**:
- ✅ Action F1: 81.6% 달성 (P5)
- ✅ 재료 매칭: 45.6% 달성 (P6)
- ✅ Grounding Rate: 64.3% (P1 - 가장 신뢰할 수 있음)
- ⚠️ Trade-off: 더 상세할수록 Hallucination 증가

---

## 🚀 시작하기

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/BioProtocolAgent.git
cd BioProtocolAgent

# 가상환경 생성
python3.12 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 입력
```

### 빠른 시작

```python
from src.data_loader import make_initial_state
from src.graph_builder import build_graph

# 그래프 생성
graph = build_graph()

# 초기 상태 설정
protocol_id = "Bio-protocol-2302"
init_state = make_initial_state(protocol_id)

# 실행
final_state = graph.invoke(init_state)

# 결과 확인
print(f"Protocol: {final_state['protocol_id']}")
print(f"생성된 액션 수: {len(final_state['actions'])}")
for action in final_state['actions']:
    print(f"  - {action['action']}: {action['description']}")
```

### 평가 실행

```bash
# 1. Action Extraction 평가 (Ablation Study)
python scripts/eval_ablation_actions_v2.py

# 2. 자연어 프로토콜 비교 (BLEU, ROUGE, F1)
python scripts/eval_hier_protocol_vs_generated.py

# 3. Grounding 및 Hallucination 평가 (개선됨)
python scripts/eval_methods_grounding_v2.py

# 4. 결과 확인
# - data/ablation_eval_actions_v2.csv
# - reports/grounding_eval/per_protocol_grounding_v2.csv
# - reports/grounding_eval/GROUNDING_ANALYSIS.md
```

---

## 📁 프로젝트 구조

```
BioProtocolAgent/
├── README.md                         # 이 파일
├── QUICKSTART.md                     # 5분 설치 가이드
├── STRUCTURE.md                      # 디렉토리 구조 상세
├── requirements.txt                  # 파이썬 의존성
├── .env.example                      # 환경변수 템플릿
├── .gitignore
├── main.py                           # 엔트리 포인트
│
├── src/                              # 핵심 소스 코드
│   ├── types.py                      # TypedDict 정의
│   ├── graph_builder.py              # LangGraph 구성
│   ├── data_loader.py                # JSONL 데이터 로드
│   ├── nodes/                        # 에이전트 노드들
│   │   ├── task_planner.py
│   │   ├── step_planner.py
│   │   ├── action_extractor.py
│   │   ├── condition_extractor.py
│   │   ├── verifier.py               # CoV 검증
│   │   ├── verifier_react.py         # ReAct 검증
│   │   ├── writer.py
│   │   └── order_structurer.py
│   ├── tools/                        # 도구 레이어
│   │   ├── rag_search.py
│   │   ├── verifier_tools.py
│   │   └── ...
│   └── ...
│
├── scripts/                          # 평가 스크립트
│   ├── eval_ablation_actions_v2.py   # ✅ Action 추출 평가
│   ├── eval_hier_protocol_vs_generated.py  # ✅ 자연어 프로토콜 평가
│   ├── eval_methods_grounding_v2.py  # ✅ Grounding & Hallucination 평가
│   ├── gen_protocols_from_actions_llm.py
│   └── run_ablation_generation.py
│
├── data/                             # 데이터셋
│   ├── gold_pairs_testset_v2.jsonl   # 테스트셋
│   ├── gold_actions_ir_10.jsonl      # Gold 액션
│   ├── gen_actions_ir_10.jsonl       # 생성된 액션
│   └── ablation/                     # 생성 결과
│
├── reports/                          # 평가 결과
│   ├── grounding_eval/               # ✅ Grounding 평가
│   │   ├── per_protocol_grounding_v2.csv
│   │   ├── summary_modes_grounding_v2.csv
│   │   └── GROUNDING_ANALYSIS.md     # 📋 상세 분석
│   ├── llm_protocols/                # 생성된 프로토콜
│   └── ...
│
├── _archive/                         # 이전 실험 (300 MB)
└── BioProtocol_Interview_Report.pdf  # 기술 문서
```

---

## 🔧 기술 스택

### 핵심 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **LangGraph** | 1.0.4 | 멀티에이전트 오케스트레이션 |
| **LangChain** | 1.1.0 | LLM 인터페이스 |
| **OpenAI** | 2.6.0 | GPT-4, GPT-4o-mini API |
| **Pydantic** | 2.12.3 | 스키마 검증 |
| **Sentence-Transformers** | 5.1.2 | 문장 임베딩 |
| **FAISS** | 1.12.0 | 벡터 검색 |
| **Torch** | 2.9.0 | 신경망 계산 |
| **Pandas** | 최신 | 데이터 분석 |

### 평가 라이브러리
- **SacreBleu**: BLEU 스코어
- **ROUGE**: 문장 유사도
- **scikit-learn**: TF-IDF, 코사인 유사도

### 런타임
- **Python 3.12.7**
- **가상환경**: `.venv`

---

## 💡 핵심 기술 특징

### 1. **계층적 분해 (Hierarchical Decomposition)**
```
Protocol → Task → Step → Action
```
각 레벨에서 다른 LLM 프롬프트로 정밀도 향상

### 2. **Chain-of-Verification (CoV)**
- 검증 질문 명시적 생성
- 단계별 추론 기록 (reasoning traces)
- 로컬/글로벌 판정 분리

### 3. **Grounding 검증 (개선됨)**
- Methods 텍스트의 다중 청킹 전략 (문장 + 문자 기반)
- 생성된 프로토콜에서 다중 소스의 문장 추출
- 적절한 임계값 범위 (0.55, 0.60, 0.65)
- 상세한 통계 (평균/중간값 유사도 포함)

### 4. **벡터 기반 매칭**
- Sentence-Transformer: 문장 임베딩
- FAISS: 대규모 유사도 검색 (O(n) → O(log n))
- Soft IOU: 정확한 문자 매칭 대신 의미 기반 유사도

### 5. **에러 처리 & 로깅**
```python
# JSON 파싱 실패 시 자동 로깅
if json_parse_failed:
    log_to("src/logs/step_structurer_json_error.log")
```

---

## 📋 Grounding Evaluation 개선사항

### 🐛 이전 코드의 문제

원본 `eval_methods_grounding_protocols.py`:
```python
# ❌ 문제 1: 생성 프로토콜에서 문장을 못 찾음
pred_sents = rec.get("sentences", []) or []  # 항상 빈 리스트!

# ❌ 문제 2: 청킹이 너무 큼 (1200자)
# 결과: 대부분 Grounding rate = 0

# ❌ 문제 3: 기본 임계값이 너무 높음 (0.60)
```

### ✅ 개선사항 (`eval_methods_grounding_v2.py`)

1. **다중 소스 문장 추출**
   ```python
   - ActionIR의 description
   - protocol_text 필드
   - 모든 문자열 필드 재귀 수집
   ```

2. **이중 청킹 전략**
   ```python
   - 문장 기반: chunk_text_by_sentences()
   - 문자 기반: chunk_text_by_chars()
   - 결합: 두 방식의 장점 활용
   ```

3. **임계값 비교**
   ```python
   - 0.55: 관대 (false positive 많음)
   - 0.60: 균형잡힘 (권장)
   - 0.65: 엄격 (false negative 많음)
   ```

4. **상세 통계**
   ```python
   - Average max similarity
   - Median max similarity
   - Min/Max similarity
   ```

### 📊 결과 해석

**Hallucination이 높은 것이 항상 나쁜가?**

❌ 아님! 두 가지 가능성:

1. **실제 환각**: LLM이 Methods에 없는 내용 추가
2. **의미적 추론**: Methods의 내용에서 합리적인 추론 (전문가 수준)

더 자세한 해석은 `reports/grounding_eval/GROUNDING_ANALYSIS.md` 참조

---

## 🎓 면접 강조 포인트

### 기술적 관점
1. **LangGraph 멀티에이전트 오케스트레이션**
   - Stateful workflow (GraphState)
   - 조건부 라우팅 및 상태 검포인트

2. **LLM 프롬프트 엔지니어링**
   - JSON schema 기반 구조화된 출력
   - Temperature 조절로 결정성/탐색성 제어

3. **벡터 임베딩 & RAG**
   - Sentence-Transformers로 의미 기반 검색
   - FAISS로 대규모 검색 최적화

4. **다층적 평가 메트릭**
   - Action 추출: Precision, Recall, F1
   - 속성 매칭: Material/Condition IOU
   - 신뢰성: Grounding rate, Hallucination rate

5. **문제 해결 능력**
   - Grounding 평가 코드 버그 진단 및 수정
   - 문장 추출 전략 개선
   - 청킹 방식 최적화

### 비즈니스/연구 관점
- **과학 논문 자동화**: Methods → 재현 가능한 프로토콜
- **실험 재현성**: 근거 추적 (evidence_span)
- **신뢰성**: Chain-of-Verification으로 검증
- **정량 평가**: 설명 가능한 메트릭

---

## 📚 의존성

```
langgraph==1.0.4
langchain==1.1.0
openai==2.6.0
pydantic==2.12.3
sentence-transformers==5.1.2
faiss-cpu==1.12.0
torch==2.9.0
transformers==4.57.1
pandas>=2.0.0
scikit-learn>=1.3.0
```

설치:
```bash
pip install -r requirements.txt
```

---

## 🔗 관련 자료

- **빠른 시작**: [`QUICKSTART.md`](./QUICKSTART.md)
- **구조 설명**: [`STRUCTURE.md`](./STRUCTURE.md)
- **평가 메트릭**: [`EVALUATION_METRICS.md`](./EVALUATION_METRICS.md) ⭐ **필독!**
  - P1-P6 프롬프트 변이체 상세 설명
  - 각 평가 메트릭 계산 방식
  - CSV 파일 구조
  - 해석 가이드
- **기술 문서**: [`BioProtocol_Interview_Report.pdf`](./BioProtocol_Interview_Report.pdf)
- **Grounding 분석**: [`reports/grounding_eval/GROUNDING_ANALYSIS.md`](./reports/grounding_eval/GROUNDING_ANALYSIS.md)
- **아카이브**: [`_archive/`](./_archive/) - 이전 실험

---

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

## 👤 저자

**Haebin Kim** (nice2pinky@gmail.com)

---

**마지막 업데이트**: 2026-07-06  
**상태**: 논문 초고 완성 + Grounding 평가 개선 ✅
