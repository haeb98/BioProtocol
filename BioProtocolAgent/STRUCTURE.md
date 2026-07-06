# 📁 프로젝트 구조 설명

## 디렉토리 계층

### 루트 레벨
```
BioProtocolAgent/
├── README.md                          # GitHub 메인 문서 (필독!)
├── QUICKSTART.md                      # 빠른 시작 가이드
├── STRUCTURE.md                       # 이 파일
├── requirements.txt                   # 파이썬 의존성
├── .env.example                       # 환경변수 템플릿
├── main.py                            # 엔트리 포인트
└── .gitignore                         # Git 무시 파일
```

---

## 📂 src/ - 핵심 소스 코드

### src/types.py
**타입 정의 (TypedDict)**
- `GraphState`: 전체 워크플로우 상태
- `ActionIR`: 액션 표현 (action, materials, conditions, produces)
- `StepIR`: 단계 표현
- `ParameterIR`, `MaterialIR`, `ConditionIR`: 세부 속성

### src/graph_builder.py
**LangGraph 구성**
```python
g = StateGraph(GraphState)
g.add_node("task_planner", task_planner_node)
g.add_node("step_planner", step_planner_node)
g.add_node("action_extractor", action_extractor_node)
g.add_node("condition_extractor", condition_extractor_node)
# ...
return g.compile()
```

### src/data_loader.py
**데이터 로드**
- `load_pairs_index()`: JSONL 파일 로드
- `make_initial_state()`: 초기 GraphState 생성

### src/nodes/ - 에이전트 노드들

| 파일 | 역할 | LLM 모델 |
|------|------|---------|
| **task_planner.py** | Methods에서 Task 추출 | GPT-4-1106 |
| **step_planner.py** | Task를 Step으로 구조화 | GPT-4o-mini |
| **action_extractor.py** | Step을 Action으로 분해 | GPT-4o-mini |
| **condition_extractor.py** | 조건(온도, 시간 등) 추출 | GPT-4o-mini |
| **verifier.py** | Chain-of-Verification 검증 | GPT-4o-mini |
| **verifier_react.py** | ReAct 스타일 검증 | GPT-4o-mini |
| **writer.py** | ActionIR → 자연어 프로토콜 | GPT-4o-mini |
| **order_structurer.py** | 액션 순서 최적화 | GPT-4o-mini |

### src/tools/ - 도구 레이어

| 파일 | 역할 |
|------|------|
| **rag_search.py** | FAISS 기반 벡터 검색 (Methods 근거 찾기) |
| **doc_search.py** | 문서 기반 검색 |
| **verifier_tools.py** | 검증 중 보조 도구 |
| **calculate_tool.py** | 농도, 부피 계산 |
| **reorder_tool.py** | 액션 순서 재정렬 |

### src/eval/ - 평가 유틸
- 평가 메트릭 계산
- 성능 분석 도구

### src/utils/ - 헬퍼 함수
- 텍스트 전처리
- 파싱 유틸

### src/logs/ - 에러 로깅
```
step_structurer_json_error.log  # JSON 파싱 실패 로그
```

---

## 📊 scripts/ - 실험 & 평가 스크립트

### 핵심 스크립트

#### 1. eval_ablation_actions_v2.py
**Ablation Study - 액션 IR 평가 (최종 버전)**

기능:
- Gold Action IR vs Generated Action IR 비교
- Step-level 매칭 (Sentence-Transformer, threshold=0.7)
- Material/Condition Soft IOU 계산
- Methods 근거 검증 (chunk 기반)
- 액션 순서 일관성 검사

입력: `data/gold_actions_ir_10.jsonl`, `data/ablation/`
출력: `data/ablation_eval_actions_v2.csv`

메트릭:
```
step_precision, step_recall, step_f1
order_score, mat_iou, cond_iou
grounding_hallucination_rate, evidence_coverage
```

#### 2. eval_hier_protocol_vs_generated.py
**자연어 프로토콜 평가**

기능:
- 생성된 프로토콜 vs Gold 프로토콜 비교
- BLEU, ROUGE, TF-IDF 코사인 유사도 계산
- Step-level F1 (임베딩 기반)

메트릭:
```
bleu_score, rouge_score
cosine_similarity, step_f1
```

#### 3. eval_methods_grounding_protocols.py
**Grounding 검증**

기능:
- 생성된 액션이 Methods 텍스트에 근거를 가지는지 확인
- 청크 단위 유사도 계산 (chunk_size=1200, overlap=200)
- 환각 탐지

메트릭:
```
grounding_score, hallucination_rate
evidence_coverage
```

#### 4. gen_protocols_from_actions_llm.py
**ActionIR → 자연어 프로토콜 변환**

기능:
- ActionIR을 입력으로 받아서
- Writer 노드를 사용하여 자연어 프로토콜 생성
- 계층적 구조 유지

#### 5. run_ablation_generation.py
**전체 파이프라인 실행**

기능:
- 테스트셋의 모든 프로토콜에 대해
- Task Planner → Step Planner → Action Extractor → ... 실행
- 결과를 `data/ablation/`에 저장

---

## 📁 data/ - 데이터셋

### 핵심 데이터

#### gold_pairs_testset_v2.jsonl
**테스트셋 (10개 프로토콜)**

구조:
```json
{
    "protocol_id": "Bio-protocol-2302",
    "bio": { "title": "RNA Extraction...", ... },
    "article": { "title": "...", "pmcid": "...", ... },
    "sec_text": "Methods section text...",
    "hierarchical_protocol": { ... },
    "pmcid": "PMC1234567"
}
```

#### gold_actions_ir_10.jsonl
**Gold 액션 IR (수동 주석)**

프로토콜당 평균 40-70개 액션
- 정밀한 재료/조건 정보
- Methods에서의 정확한 위치 추적

#### gen_actions_ir_10.jsonl
**생성된 액션 IR (모델 출력)**

사용 사례: Baseline 비교

#### ablation/
**Ablation Study 결과**

프롬프트 variant별 생성 결과:
```
P1_gen_actions.jsonl  (기본 프롬프트)
P2_gen_actions.jsonl  (간단한 버전)
P3_gen_actions.jsonl  (다중 선택)
...
P6_gen_actions.jsonl  (최적화 버전)
```

---

## 📈 reports/ - 평가 결과

### CSV 파일들

#### ablation_eval_actions_v2.csv
**메인 평가 결과**

컬럼:
```
protocol_id, n_gold, n_pred
step_precision, step_recall, step_f1
order_score, mat_iou, cond_iou
grounding_hallucination_rate, evidence_coverage
mode (P1-P6), has_methods_text
```

#### ablation_eval_actions_v2_summary.csv
**요약 통계**

프롬프트 variant별 평균:
```
P1: Step F1=0.494, Mat IOU=0.247, Hallucination=0.25
P2: Step F1=0.317, Mat IOU=0.318, Hallucination=0.00
...
P6: Step F1=0.741, Mat IOU=0.456, Hallucination=0.00
```

### 서브디렉토리

#### grounding_eval/
**Methods 근거 검증 결과**
- 각 액션별 grounding score

#### llm_protocols/
**생성된 자연어 프로토콜**
- 최종 프로토콜 텍스트 (.txt 또는 .json)

---

## 📔 notebooks/ - 시각화 & 실험

주피터 노트북 (선택적)
- 결과 시각화
- 인터랙티브 분석

---

## 🗂️ _archive/ - 이전 실험 아카이브

프로젝트 정리로 옮겨진 파일들:

### _archive/scripts/
```
eval_action_soft_iou.py          (중복)
eval_action_soft_iou_10.py       (중복)
eval_hallucination_metrics.py    (테스트용)
test_doc_search.py               (테스트용)
gen_gen_actions_ir_10.py         (데이터 생성용)
gen_gold_actions_ir_10.py        (데이터 생성용)
```

### _archive/data/
```
ablation_eval_actions_all.csv         (구버전)
ablation_eval_actions_all_add.csv     (구버전)
eval_actions_soft_iou_10.csv          (중복)
gen_steps_A.jsonl                     (구버전)
gen_steps_B.jsonl                     (구버전)
```

### _archive/reports/
```
protocol_eval/          (구버전 평가)
protocol_eval_0.8/      (구버전 평가)
protocol_eval_0.85/     (구버전 평가)
```

---

## 📄 문서 파일

| 파일 | 용도 |
|------|------|
| **README.md** | GitHub 메인 문서 (필독!) |
| **QUICKSTART.md** | 5분 안에 시작하기 |
| **STRUCTURE.md** | 이 파일 - 구조 설명 |
| **.env.example** | 환경변수 템플릿 |
| **BioProtocol_Interview_Report.pdf** | 기술 문서 (면접용) |
| **BioProtocol_Interview_Report.html** | 웹 버전 기술 문서 |

---

## 🔄 데이터 흐름

```
[gold_pairs_testset_v2.jsonl]
        ↓
    [main.py]
        ↓
[Task Planner] ──→ [Tasks]
        ↓
[Step Planner] ──→ [StepIR]
        ↓
[Action Extractor] ──→ [ActionIR]
        ↓
[Condition Extractor] ──→ [ConditionIR]
        ↓
[Verifier] ──→ [Verified ActionIR]
        ↓
[Writer] ──→ [Natural Language Protocol]
        ↓
[gen_actions_ir_10.jsonl] ──→ [평가 스크립트]
                                    ↓
                        [ablation_eval_actions_v2.csv]
```

---

## 💾 파일 크기 추정

```
src/                    ~100 KB
  nodes/               ~50 KB
  tools/               ~30 KB
  
scripts/               ~150 KB

data/                  ~500 MB (JSONL 데이터)
  gold_*.jsonl        ~50 MB
  ablation/           ~450 MB

reports/               ~100 MB

_archive/              ~300 MB (이전 실험)

README & docs          ~200 KB
```

---

## 🔍 주요 파일 찾기

**"Task 추출을 보고 싶은데?"**
→ `src/nodes/task_planner.py`

**"액션을 어떻게 검증하는데?"**
→ `src/nodes/verifier.py` 또는 `src/nodes/verifier_react.py`

**"성능 지표가 뭔데?"**
→ `scripts/eval_ablation_actions_v2.py`

**"자연어 프로토콜 어떻게 생성하는데?"**
→ `src/nodes/writer.py` 또는 `scripts/gen_protocols_from_actions_llm.py`

**"내 데이터 어디 넣는데?"**
→ `data/` 폴더에 JSONL 형식으로 추가

**"에러 메시지 어디 보는데?"**
→ `src/logs/step_structurer_json_error.log`

---

## ✅ 체크리스트

프로젝트 설정 확인:
- [ ] README.md 읽음
- [ ] QUICKSTART.md로 설치 완료
- [ ] `.env` 파일에 OpenAI API 키 입력
- [ ] `main.py` 실행 성공
- [ ] 평가 스크립트 실행 성공
- [ ] 결과 CSV 파일 생성됨

---

**마지막 업데이트**: 2025년 12월 15일
