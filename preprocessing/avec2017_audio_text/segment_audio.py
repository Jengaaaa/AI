import os
import soundfile as sf
import numpy as np

AUDIO_DIR = "data/avec2017/processed/audio_16k"
OUT_DIR = "data/avec2017/processed/segments_audio"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SEGMENTS = 50

def segment_audio(subject_id):
    in_path = os.path.join(AUDIO_DIR, f"{subject_id}.wav")
    out_dir = os.path.join(OUT_DIR, subject_id)
    os.makedirs(out_dir, exist_ok=True)

    # librosa 대신 soundfile 사용 → 파형 보존
    audio, sr = sf.read(in_path)

    # 모노 강제 변환
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    total_len = len(audio)
    seg_len = total_len // NUM_SEGMENTS

    for i in range(NUM_SEGMENTS):
        start = i * seg_len
        end = start + seg_len if i < NUM_SEGMENTS-1 else total_len
        seg = audio[start:end]

        # 너무 짧으면 zero padding (논문 방식)
        if len(seg) < seg_len:
            pad = seg_len - len(seg)
            seg = np.pad(seg, (0, pad), mode="constant")

        out_path = os.path.join(out_dir, f"seg_{i:03d}.wav")
        sf.write(out_path, seg, sr)

    print(f"[OK] {subject_id} 오디오 세그먼트 생성 완료")

if __name__ == "__main__":
    subjects = [
        f[:-4] for f in os.listdir(AUDIO_DIR)
        if f.endswith(".wav")
    ]

    for subject in sorted(subjects):
        segment_audio(subject)
