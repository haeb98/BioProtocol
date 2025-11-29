import json
import shutil
from pathlib import Path

# 임시 저장된 결과 파일 경로 (필요시 수정)
tmp_path = "b2_steps_new.jsonl"
output_path = "runs/b2_steps_new.jsonl"

# 1. 임시 결과 불러오기
with open(tmp_path, "r", encoding="utf-8") as f:
    all_results = [json.loads(line) for line in f if line.strip()]

# 2. 기존에 디렉토리로 잘못 생성된 경우 삭제
if Path(output_path).is_dir():
    shutil.rmtree(output_path)

# 3. 정식 결과 파일로 다시 저장
with open(output_path, "w", encoding="utf-8") as f:
    for item in all_results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 저장 완료: {output_path}, 총 {len(all_results)}개")
