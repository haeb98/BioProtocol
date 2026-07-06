# 🚀 빠른 시작 가이드

## 설치 (5분)

### 1단계: 저장소 클론

```bash
cd /path/to/workspace
git clone https://github.com/yourusername/BioProtocolAgent.git
cd BioProtocolAgent
```

### 2단계: 가상환경 설정

```bash
python3.12 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 3단계: 의존성 설치

```bash
pip install -r requirements.txt
```

### 4단계: 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 입력
```

---

## 사용 방법

### 방법 1: Python 스크립트 사용 (추천)

```python
from src.data_loader import make_initial_state
from src.graph_builder import build_graph

# 그래프 생성
graph = build_graph()

# 프로토콜 ID (테스트셋에 포함된 ID 사용)
protocol_id = "Bio-protocol-2302"

# 초기 상태 생성
init_state = make_initial_state(protocol_id)

# 파이프라인 실행
final_state = graph.invoke(init_state)

# 결과 확인
print(f"Protocol ID: {final_state['protocol_id']}")
print(f"생성된 액션 수: {len(final_state['actions'])}")

for i, action in enumerate(final_state['actions']):
    print(f"\n[액션 {i + 1}]")
    print(f"  타입: {action['action']}")
    print(f"  설명: {action['description']}")
    print(f"  재료: {action.get('materials', [])}")
    print(f"  조건: {action.get('conditions', [])}")
    print(f"  결과: {action.get('produces', [])}")
```

### 방법 2: main.py 실행

```bash
python main.py
```

출력 예:

```
=== Protocol ID ===
Bio-protocol-2302
RNA Extraction Protocol

== Bio-protocol-2302::T1::S1::A1 ==
ACTION : Add
DESC   : Add RNA sample to lysis buffer
MATS   : [{'name': 'RNA sample', 'role': 'substrate', 'volume': '100 μL'}, ...]
CONDS  : [{'type': 'temperature', 'value': 'room temperature'}]
PROD   : ['lysed RNA sample']
EVID   : "Add the RNA sample to lysis buffer and..."
...
```

---

## 평가 실행

### 1. Ablation Study (액션 IR 평가)

```bash
python scripts/eval_ablation_actions_v2.py
```

출력: `data/ablation_eval_actions_v2.csv`

- Step Precision/Recall/F1
- Material/Condition IOU
- Grounding hallucination rate

### 2. 자연어 프로토콜 평가

```bash
python scripts/eval_hier_protocol_vs_generated.py
```

평가 메트릭:

- BLEU score
- ROUGE score
- Step-level F1

### 3. Methods 근거 검증

```bash
python scripts/eval_methods_grounding_protocols.py
```

출력: Grounding 신뢰도 점수

---

## 데이터 형식

### 입력: JSONL 형식

**금고_pairs_testset_v2.jsonl**

```json
{
  "protocol_id": "Bio-protocol-2302",
  "bio": {
    "title": "RNA Extraction Protocol",
    "description": "..."
  },
  "article": {
    "title": "A new approach to RNA extraction",
    "pmcid": "PMC1234567",
    ...
  },
  "sec_text": "Methods section from the paper...",
  "hierarchical_protocol": {
    ...
  }
}
```

### 출력: ActionIR 형식

```json
{
  "action_id": "Bio-protocol-2302::T1::S1::A1",
  "action": "Add",
  "description": "Add RNA sample to lysis buffer",
  "materials": [
    {
      "name": "RNA sample",
      "role": "substrate",
      "volume": "100 μL"
    }
  ],
  "conditions": [
    {
      "type": "temperature",
      "value": "room temperature"
    }
  ],
  "produces": [
    "lysed RNA sample"
  ],
  "evidence_span": "Add the RNA sample to lysis buffer and..."
}
```

---

## 아키텍처 흐름

```
[Methods Text] → [Task Planner]
                      ↓
                [Step Planner]
                      ↓
            [Action Extractor]
                      ↓
          [Condition Extractor]
                      ↓
            [Verifier (CoV)]
                      ↓
                 [Writer]
                      ↓
    [ActionIR + Natural Protocol]
```

각 단계:

1. **Task Planner**: Methods에서 핵심 작업 추출
2. **Step Planner**: 작업을 세부 단계로 구분
3. **Action Extractor**: 단계를 원자적 액션으로 분해
4. **Condition Extractor**: 온도, 시간 등 조건 추출
5. **Verifier**: Chain-of-Verification으로 검증
6. **Writer**: 최종 자연어 프로토콜 생성

---

## 문제 해결

### 1. OpenAI API 키 오류

```
"AuthenticationError: No API key provided"
```

→ `.env` 파일에 유효한 OPENAI_API_KEY 입력 확인

### 2. JSON 파싱 오류

```
"JSONDecodeError: Expecting value"
```

→ 로그 확인: `src/logs/step_structurer_json_error.log`
→ 프롬프트 또는 모델 변경 고려

### 3. 메모리 부족

```
"RuntimeError: CUDA out of memory"
```

→ `faiss-cpu` 사용 (현재 설정)
→ 배치 사이즈 감소

---

## 테스트셋 프로토콜 ID

```
Bio-protocol-1111   (68개 액션)
Bio-protocol-1373   (72개 액션)
Bio-protocol-2096   (39개 액션)
Bio-protocol-2302   (24개 액션) ← 추천
Bio-protocol-2617   (40개 액션)
Bio-protocol-3584   (52개 액션)
Bio-protocol-3607   (41개 액션)
Bio-protocol-3617   (26개 액션)
Bio-protocol-851    (61개 액션)
Bio-protocol-972    (55개 액션)
```

---

## 다음 단계

1. **커스텀 프로토콜 추가**
    - `data/custom_pairs.jsonl` 생성
    - `make_initial_state()` 수정

2. **프롬프트 최적화**
    - `src/nodes/task_planner.py` 수정
    - `src/nodes/step_planner.py` 수정

3. **새로운 도구 추가**
    - `src/tools/` 디렉토리에 새 도구 생성
    - `graph_builder.py`에 등록

---

## 추가 정보

- 기술 문서: [`BioProtocol_Interview_Report.pdf`](./BioProtocol_Interview_Report.pdf)
- 전체 README: [`README.md`](../README.md)
- 아카이브: [`_archive/`](./_archive/)

---

**행운을 빕니다! 🎉**
