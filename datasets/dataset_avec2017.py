import os
import torch
import librosa
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class AVEC2017AudioTextDataset(Dataset):
    def __init__(self, 
                 root="data/avec2017/processed",
                 split_file="data/avec2017/raw/train_split.csv",
                 tokenizer_name="klue/bert-base",
                 num_segments=50,
                 max_text_len=64):

        self.root = root
        self.num_segments = num_segments
        self.max_text_len = max_text_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load split list
        self.data = []
        with open(split_file, "r") as f:
            lines = f.readlines()[1:]  # skip header

        for line in lines:
            subject, label = line.strip().split(",")

            # subject = '300' -> '300_P'
            if not subject.endswith("_P"):
                subject = subject + "_P"

            self.data.append((subject, float(label)))

    def __len__(self):
        return len(self.data)

    def load_audio_segments(self, subject_id):
        seg_dir = f"{self.root}/segments_audio/{subject_id}"
        audio_tensors = []

        for i in range(self.num_segments):
            seg_path = f"{seg_dir}/seg_{i:03d}.wav"
            audio, sr = librosa.load(seg_path, sr=16000)
            audio_tensors.append(torch.tensor(audio, dtype=torch.float32))

        max_len = max([a.shape[0] for a in audio_tensors])
        padded = [torch.nn.functional.pad(a, (0, max_len - len(a))) for a in audio_tensors]

        return torch.stack(padded)   # (S, audio_len)

    def load_text_segments(self, subject_id):
        seg_dir = f"{self.root}/segments_text/{subject_id}"
        text_tensors = []

        for i in range(self.num_segments):
            seg_path = f"{seg_dir}/seg_{i:03d}.txt"

            if os.path.exists(seg_path):
                with open(seg_path, "r") as f:
                    text = " ".join([l.strip() for l in f.readlines()])
            else:
                text = ""

            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt"
            )

            text_tensors.append(encoded["input_ids"].squeeze(0))

        return torch.stack(text_tensors)

    def __getitem__(self, idx):
        subject_id, label = self.data[idx]

        audio = self.load_audio_segments(subject_id)
        text = self.load_text_segments(subject_id)

        return {
            "audio": audio,
            "text": text,
            "label": torch.tensor(label, dtype=torch.float32)
        }
