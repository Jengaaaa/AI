import os
import librosa
import soundfile as sf

RAW_DIR = "data/avec2017/raw"
OUT_DIR = "data/avec2017/processed/audio_16k"
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess_audio(subject_id):
    in_path = f"{RAW_DIR}/{subject_id}/audio.wav"
    out_path = f"{OUT_DIR}/{subject_id}.wav"

    audio, sr = librosa.load(in_path, sr=16000, mono=True)
    sf.write(out_path, audio, 16000)
    print(f"[OK] {subject_id} 오디오 저장 완료 → {out_path}")

if __name__ == "__main__":
    subjects = sorted(os.listdir(RAW_DIR))
    for subject in subjects:
        preprocess_audio(subject)
