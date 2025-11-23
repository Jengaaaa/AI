import os

TXT_DIR = "data/avec2017/processed/transcript_clean"
OUT_DIR = "data/avec2017/processed/segments_text"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SEGMENTS = 50

def segment_text(subject_id):
    in_path = f"{TXT_DIR}/{subject_id}.txt"
    out_dir = f"{OUT_DIR}/{subject_id}"
    os.makedirs(out_dir, exist_ok=True)

    # 문장 읽기
    with open(in_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # 분배
    per_seg = max(1, len(lines) // NUM_SEGMENTS)

    for i in range(NUM_SEGMENTS):
        start = i * per_seg
        end = (i + 1) * per_seg
        seg_lines = lines[start:end]

        with open(f"{out_dir}/seg_{i:03d}.txt", "w") as f_out:
            for l in seg_lines:
                f_out.write(l + "\n")

    print(f"[OK] {subject_id} 텍스트 세그먼트 생성 완료")

if __name__ == "__main__":
    subjects = sorted(os.listdir(TXT_DIR))
    subjects = [s.replace(".txt", "") for s in subjects]

    for subject in subjects:
        segment_text(subject)
