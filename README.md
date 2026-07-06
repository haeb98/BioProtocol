# BioProtocolAgent: 멀티에이전트 기반 바이오 프로토콜 자동 생성

> 과학 논문의 Methods 섹션에서 자동으로 **재현 가능한 바이오 실험 프로토콜**을 생성하는 멀티 에이전트 시스템

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

| 노드                      | 역할                      | LLM         | 입력               | 출력               |
|-------------------------|-------------------------|-------------|------------------|------------------|
| **Task Planner**        | Methods에서 핵심 Task 추출    | GPT-4-1106  | Methods, 제목      | Task 리스트         |
| **Step Planner**        | Task를 세부 Step으로 구조화     | GPT-4o-mini | Task + Methods   | StepIR           |
| **Action Extractor**    | Step을 원자적 Action으로 분해   | GPT-4o-mini | Step + Methods   | ActionIR         |
| **Condition Extractor** | 물리적 조건(온도, 시간 등) 추출     | GPT-4o-mini | Action           | ConditionIR      |
| **Verifier**            | 생성된 내용을 Methods 텍스트로 검증 | GPT-4o-mini | Action + Methods | Verification 결과  |
| **Writer**              | ActionIR을 자연어 프로토콜로 변환  | GPT-4o-mini | ActionIR         | Natural protocol |

---

## 📊 성능 결과

### 평가 메트릭

- **Step F1**: 단계 추출 정확도 (0-1)
- **Material IOU**: 재료 매칭 (Soft Intersection-over-Union)
- **Condition IOU**: 조건 매칭 (온도, 시간 등)
- **Grounding**: Methods 텍스트 근거 검증

### 최고 성능 (Prompt Variant P5, P6)

| 메트릭               | P1 (기본) | P5 (다중회전) | P6 (최적)   |
|-------------------|---------|-----------|-----------|
| **Step F1**       | 0.494   | **0.816** | **0.741** |
| **Material IOU**  | 0.247   | **0.415** | **0.456** |
| **Condition IOU** | 0.573   | 0.531     | 0.532     |

**핵심 발견사항**:

- 다중 회전 및 개선된 프롬프트로 **Step F1 81.6% 달성**
- Methods 근거 있는 추출로 **환각 0% 달성**
- 조건 추출이 특히 강력 (IOU 53-68%)

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
echo "OPENAI_API_KEY=your-api-key-here" > .env
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
# Ablation Study (액션 IR 평가)
python scripts/eval_ablation_actions_v2.py

# 계층적 프로토콜 비교 (자연어 평가)
python scripts/eval_hier_protocol_vs_generated.py

# Grounding 검증 (Methods 근거 확인)
python scripts/eval_methods_grounding_protocols.py
```

---

## 📁 프로젝트 구조

```
BioProtocolAgent/
├── README.md                         # 이 파일
├── requirements.txt                  # 의존성
├── .env.example                      # 환경변수 템플릿
├── main.py                           # 엔트리 포인트
│
├── src/                              # 핵심 소스 코드
│   ├── types.py                      # TypedDict 정의 (GraphState, ActionIR 등)
│   ├── graph_builder.py              # LangGraph 구성
│   ├── data_loader.py                # JSONL 데이터 로드
│   ├── nodes/                        # 에이전트 노드들
│   │   ├── task_planner.py           # Task 추출
│   │   ├── step_planner.py           # Step 구조화
│   │   ├── action_extractor.py       # Action 분해
│   │   ├── condition_extractor.py    # 조건 추출
│   │   ├── verifier.py               # CoV 검증
│   │   ├── verifier_react.py         # ReAct 검증
│   │   ├── writer.py                 # 자연어 생성
│   │   └── order_structurer.py       # 순서 최적화
│   ├── tools/                        # 도구 레이어
│   │   ├── rag_search.py             # FAISS 기반 벡터 검색
│   │   ├── verifier_tools.py         # 검증 도구
│   │   └── ...
│   ├── eval/                         # 평가 유틸
│   ├── utils/                        # 헬퍼 함수
│   └── logs/                         # 에러 로깅
│
├── scripts/                          # 실험 & 평가 스크립트
│   ├── eval_ablation_actions_v2.py   # Ablation 평가 (최종)
│   ├── eval_hier_protocol_vs_generated.py  # 자연어 프로토콜 평가
│   ├── eval_methods_grounding_protocols.py # Grounding 검증
│   ├── gen_protocols_from_actions_llm.py   # IR → 프로토콜 변환
│   └── run_ablation_generation.py          # 전체 파이프라인 실행
│
├── data/                             # 데이터셋
│   ├── gold_pairs_testset_v2.jsonl   # 테스트셋 (논문+Methods)
│   ├── gold_actions_ir_10.jsonl      # Gold 액션 IR
│   ├── gen_actions_ir_10.jsonl       # 생성된 액션 IR
│   └── ablation/                     # Ablation 결과
│
├── reports/                          # 평가 결과
│   ├── grounding_eval/               # Grounding 평가 결과
│   ├── llm_protocols/                # 생성된 프로토콜
│   └── *.csv                         # 평가 메트릭
│
├── notebooks/                        # 시각화 & 실험
├── _archive/                         # 아카이브 (이전 실험)
└── BioProtocol_Interview_Report.pdf  # 기술 문서
```

---

## 🔧 기술 스택

### 핵심 라이브러리

| 라이브러리                     | 버전     | 용도                     |
|---------------------------|--------|------------------------|
| **LangGraph**             | 1.0.4  | 멀티에이전트 오케스트레이션         |
| **LangChain**             | 1.1.0  | LLM 인터페이스              |
| **OpenAI**                | 2.6.0  | GPT-4, GPT-4o-mini API |
| **Pydantic**              | 2.12.3 | 스키마 검증                 |
| **Sentence-Transformers** | 5.1.2  | 문장 임베딩                 |
| **FAISS**                 | 1.12.0 | 벡터 검색                  |
| **Torch**                 | 2.9.0  | 신경망 계산                 |
| **Pandas**                | 최신     | 데이터 분석                 |
| **NumPy**                 | 최신     | 수치 계산                  |
| **SciPy**                 | 최신     | 선형 할당 (Hungarian)      |

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

### 3. **Grounding검증**

- Methods 텍스트의 청크 단위로 의미 유사도 확인
- 근거 없는 환각 탐지 및 필터링

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

## 📊 데이터 형식

### ActionIR (실험 액션 표현)

```python
{
    "action_id": "Bio-protocol-2302::T3::S1::A1",
    "action": "Incubate",
    "description": "해당 샘플을 37°C에서 2시간 배양",
    "materials": [
        {
            "name": "RNA 샘플",
            "role": "substrate",
            "volume": "100 μL",
            "concentration": "1 mg/mL",
            "state": "on ice"
        }
    ],
    "conditions": [
        {
            "type": "temperature",
            "value": "37",
            "unit": "°C"
        },
        {
            "type": "duration",
            "value": "2",
            "unit": "h"
        }
    ],
    "produces": ["purified RNA"],
    "evidence_span": "Incubate the sample at 37°C for 2 hours...",
    "verification": {
        "global_verdict": "supported",
        "reasoning_traces": [...],
        "revision_suggestion": ""
    }
}
```

---

## 🔬 논문 & 실험

### 주요 기여

1. **멀티에이전트 프레임워크**: LangGraph를 활용한 계층적 액션 추출
2. **검증 기반 생성**: Chain-of-Verification으로 신뢰성 보장
3. **Grounding 검증**: Methods 텍스트 근거 추적
4. **다층적 평가**: 구조, 속성, 신뢰성에 대한 종합 평가

### 평가 데이터셋

- **10개 Bio-protocols** 테스트셋
- **Gold 액션 IR** 및 **자연어 프로토콜** 주석
- **Methods 텍스트** 근거 추적

### 논문 상태

- **초고**: 2025년 12월 완성
- **주요 결과**: Step F1 81.6% (P5), Material/Condition IOU 53-68%

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
    - Step 추출 (Precision, Recall, F1)
    - 속성 매칭 (Material/Condition IOU)
    - 신뢰성 (Grounding hallucination rate)

### 비즈니스 관점

- **과학 논문 자동화**: Methods → 재현 가능한 프로토콜
- **실험 재현성**: 근거 추적 (evidence_span)
- **신뢰성**: Chain-of-Verification으로 검증

---

## 📚 의존성

### `requirements.txt`

```
langgraph==1.0.4
langchain==1.1.0
langchain-core==1.1.0
openai==2.6.0
pydantic==2.12.3

sentence-transformers==5.1.2
faiss-cpu==1.12.0
torch==2.9.0
transformers==4.57.1

pandas==latest
numpy==latest
scipy==latest

python-dotenv==0.9.9
```

### 설치

```bash
pip install -r requirements.txt
```

---

## 🔗 관련 자료

- **기술 문서**: [`BioProtocol_Interview_Report.pdf`](BioProtocolAgent/BioProtocol_Interview_Report.pdf)
- **HTML 문서**: [`BioProtocol_Interview_Report.html`](BioProtocolAgent/BioProtocol_Interview_Report.html)
- **아카이브**: 이전 실험 및 중복 파일은 [`_archive/`](./_archive/) 폴더에 보관

---

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

## 👤 저자

**Haebin Kim** (nice2pinky@gmail.com)

---

**마지막 업데이트**: 2025년 12월 15일  
**상태**: 논문 초고 완성 ✅
