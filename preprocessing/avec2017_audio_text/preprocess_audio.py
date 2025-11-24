import os
import csv
import re

RAW_DIR = "data/avec2017/raw"
OUT_DIR = "data/avec2017/processed/transcript_clean"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = text.replace('"', '')
    text = re.sub(r"[^a-z0-9 <>]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_transcript(subject_id):
    # 폴더명: 300_P → base_id = 300
    base_id = subject_id.split("_")[0]

    # 파일명: 300_TRANSCRIPT.csv
    filename = f"{base_id}_TRANSCRIPT.csv"

    # 전체 경로: data/avec2017/raw/300_P/300_TRANSCRIPT.csv
    in_path = os.path.join(RAW_DIR, subject_id, filename)
    out_path = os.path.join(OUT_DIR, f"{subject_id}.txt")

    if not os.path.exists(in_path):
        print(f"[SKIP] {in_path} 없음")
        return

    # Transcript 파일은 TAB 구분자 사용
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")

        for row in reader:
            text = row.get("value", "")
            clean = clean_text(text)
            f_out.write(clean + "\n")

    print(f"[OK] {subject_id} 처리 완료 → {out_path}")

if __name__ == "__main__":
    subjects = sorted([
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ])

    for subject in subjects:
        preprocess_transcript(subject)
