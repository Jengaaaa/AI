import os
import csv
import soundfile as sf

RAW_DIR = "data/avec2017/raw"
AUDIO_DIR = "data/avec2017/processed/audio_16k"
OUT_DIR = "data/avec2017/processed/segments_text"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SEGMENTS = 50

def segment_text(subject_id):
    # 1) transcript 파일 경로 (300_P → base_id = 300)
    base_id = subject_id.split("_")[0]
    transcript_path = f"{RAW_DIR}/{subject_id}/{base_id}_TRANSCRIPT.csv"

    # 2) audio 파일 경로
    audio_path = f"{AUDIO_DIR}/{subject_id}.wav"

    # 3) output directory
    out_dir = f"{OUT_DIR}/{subject_id}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- audio duration 계산 ----
    audio, sr = sf.read(audio_path)
    total_sec = len(audio) / sr
    seg_sec = total_sec / NUM_SEGMENTS

    # 50개 segment 텍스트 리스트
    segments = [[] for _ in range(NUM_SEGMENTS)]

    # ---- transcript CSV 읽기 ----
    with open(transcript_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            try:
                start = float(row["start_time"])
            except:
                continue

            text = row["value"].strip()

            # segment 인덱스 계산
            seg_idx = int(start // seg_sec)
            if seg_idx >= NUM_SEGMENTS:
                seg_idx = NUM_SEGMENTS - 1

            # 해당 세그먼트에 문장 추가
            segments[seg_idx].append(text)

    # ---- 각 segment를 파일로 저장 ----
    for i in range(NUM_SEGMENTS):
        out_path = f"{out_dir}/seg_{i:03d}.txt"
        with open(out_path, "w", encoding="utf-8") as f_out:
            for line in segments[i]:
                f_out.write(line + "\n")

    print(f"[OK] {subject_id} 텍스트 세그먼트 생성 완료")


if __name__ == "__main__":
    raw_list = sorted([
        s for s in os.listdir(RAW_DIR)
        if s.endswith("_P")
    ])

    for subject in raw_list:
        segment_text(subject)
