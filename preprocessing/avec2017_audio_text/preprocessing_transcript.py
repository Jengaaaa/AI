import os
import re

RAW_DIR = "data/avec2017/raw"
OUT_DIR = "data/avec2017/processed/transcript_clean"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(line):
    # 소문자
    line = line.lower()
    # <laugh>, <cough> 같은 태그 제외하고 문자만 남기기
    line = re.sub(r"[^a-z0-9<> ]", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line

def preprocess_transcript(subject_id):
    in_path = f"{RAW_DIR}/{subject_id}/transcript.txt"
    out_path = f"{OUT_DIR}/{subject_id}.txt"

    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(clean_text(line) + "\n")

    print(f"[OK] {subject_id} 텍스트 클린 완료 → {out_path}")

if __name__ == "__main__":
    subjects = sorted(os.listdir(RAW_DIR))
    for subject in subjects:
        preprocess_transcript(subject)
