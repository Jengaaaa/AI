import os
import librosa
import soundfile as sf

AUDIO_DIR = "data/avec2017/processed/audio_16k"
OUT_DIR = "data/avec2017/processed/segments_audio"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_SEGMENTS = 50

def segment_audio(subject_id):
    in_path = f"{AUDIO_DIR}/{subject_id}.wav"
    out_dir = f"{OUT_DIR}/{subject_id}"
    os.makedirs(out_dir, exist_ok=True)

    audio, sr = librosa.load(in_path, sr=16000)
    seg_len = len(audio) // NUM_SEGMENTS

    for i in range(NUM_SEGMENTS):
        start = i * seg_len
        end = min((i + 1) * seg_len, len(audio))
        seg = audio[start:end]
        sf.write(f"{out_dir}/seg_{i:03d}.wav", seg, sr)

    print(f"[OK] {subject_id} 오디오 세그먼트 생성 완료")

if __name__ == "__main__":
    subjects = sorted(os.listdir(AUDIO_DIR))
    subjects = [s.replace(".wav", "") for s in subjects]

    for subject in subjects:
        segment_audio(subject)
