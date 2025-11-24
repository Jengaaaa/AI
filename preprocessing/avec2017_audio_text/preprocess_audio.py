import os
import csv
import re

# -----------------------------
# ✔ 상대경로 그대로 사용하면서도
# ✔ 절대경로 보정해서 무조건 파일을 찾게 한다
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "avec2017", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "avec2017", "processed", "transcript_clean")

os.makedirs(OUT_DIR, exist_ok=True)


def clean_text(text):
    text = text.lower()
    text = text.replace('"', '')
    text = re.sub(r"[^a-z0-9 <>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_transcript(subject_id):
    base_id = subject_id.split("_")[0]

    in_path = os.path.join(RAW_DIR, subject_id, f"{base_id}_TRANSCRIPT.csv")
    out_path = os.path.join(OUT_DIR, f"{subject_id}.txt")

    if not os.path.exists(in_path):
        print(f"[SKIP] {in_path} 없음")
        return

    # TSV 파일임 → delimiter="\t" 필수
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")

        for row in reader:
            text = row.get("value", "")
            if text is None:
                text = ""
            clean = clean_text(text)
            f_out.write(clean + "\n")

    print(f"[OK] {subject_id} 처리 완료 → {out_path}")


if __name__ == "__main__":
    subjects = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d)) and d.endswith("_P")
    ])

    for subject in subjects:
        preprocess_transcript(subject)
