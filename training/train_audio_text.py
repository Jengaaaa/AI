# training/train_audio_text.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset_avec2017 import AVEC2017AudioTextDataset
from models.audio_text.audio_net import AudioEncoder
from models.audio_text.text_net import TextEncoder
from models.audio_text.fusion_net import AudioTextDepressionModel


def rmse_loss(pred, target):
    return torch.sqrt(nn.functional.mse_loss(pred, target))


def get_dataloader(split_file, tokenizer_name, batch_size=2, num_segments=50, max_text_len=64, shuffle=True):
    dataset = AVEC2017AudioTextDataset(
        root="data/avec2017/processed",
        split_file=split_file,
        tokenizer_name=tokenizer_name,
        num_segments=num_segments,
        max_text_len=max_text_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        audio = batch["audio"].to(device)      # (B, S, L_a)
        text = batch["text"].to(device)        # (B, S, L_t)
        label = batch["label"].to(device)      # (B,)

        optimizer.zero_grad()

        pred = model(audio, text)              # (B,)
        loss = rmse_loss(pred, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * audio.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        audio = batch["audio"].to(device)
        text = batch["text"].to(device)
        label = batch["label"].to(device)

        pred = model(audio, text)
        loss = rmse_loss(pred, label)
        total_loss += loss.item() * audio.size(0)

    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    tokenizer_name = "klue/bert-base"
    num_segments = 50
    max_text_len = 64
    batch_size = 2
    num_epochs = 10
    lr = 1e-4

    train_split = "data/avec2017/raw/train_split.csv"
    dev_split = "data/avec2017/raw/dev_split.csv"

    train_loader = get_dataloader(train_split, tokenizer_name, batch_size, num_segments, max_text_len, shuffle=True)
    dev_loader = get_dataloader(dev_split, tokenizer_name, batch_size, num_segments, max_text_len, shuffle=False)

    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder(model_name=tokenizer_name, out_dim=512, use_lstm=True)
    model = AudioTextDepressionModel(
        num_segments=num_segments,
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        feat_dim=512,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dev = 1e9
    save_dir = "checkpoints/audio_text"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        dev_loss = eval_one_epoch(model, dev_loader, device)

        print(f"[Epoch {epoch:02d}] Train RMSE: {train_loss:.4f}  |  Dev RMSE: {dev_loss:.4f}")

        if dev_loss < best_dev:
            best_dev = dev_loss
            ckpt_path = os.path.join(save_dir, f"best_model_epoch{epoch:02d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Best model 업데이트! 저장 경로: {ckpt_path}")


if __name__ == "__main__":
    main()
