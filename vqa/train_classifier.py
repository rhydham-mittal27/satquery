"""Train a small MLP head on frozen CLIP image+text embeddings for RSVQA-LR.

CLIP stays frozen (see extract_features.py); this only trains the
classifier on top, so it's fast even on CPU.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CACHE_DIR = Path(__file__).resolve().parent / "cache"
MODEL_PATH = Path(__file__).resolve().parent / "vqa_head.pt"
VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"

EPOCHS = 20
LR = 1e-3
HIDDEN = 256


class VQAHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, num_classes),
        )

    def forward(self, img_feats, text_feats):
        x = torch.cat([img_feats, text_feats], dim=-1)
        return self.net(x)


def load_npz(split):
    d = np.load(CACHE_DIR / f"{split}.npz", allow_pickle=True)
    return d["img_feats"], d["text_feats"], d["answers"]


def main():
    train_img, train_text, train_ans = load_npz("train")
    val_img, val_text, val_ans = load_npz("val")

    vocab = {a: i for i, a in enumerate(sorted(set(train_ans.tolist())))}
    idx_to_answer = {i: a for a, i in vocab.items()}
    json.dump(vocab, open(VOCAB_PATH, "w"))
    num_classes = len(vocab)
    print(f"vocab size: {num_classes}")

    def to_labels(answers):
        return np.array([vocab.get(a, -1) for a in answers])

    train_labels = to_labels(train_ans)
    val_labels = to_labels(val_ans)
    val_mask = val_labels >= 0

    train_ds = TensorDataset(
        torch.tensor(train_img, dtype=torch.float32),
        torch.tensor(train_text, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    embed_dim = train_img.shape[1]
    model = VQAHead(embed_dim, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    val_img_t = torch.tensor(val_img[val_mask], dtype=torch.float32)
    val_text_t = torch.tensor(val_text[val_mask], dtype=torch.float32)
    val_labels_t = torch.tensor(val_labels[val_mask], dtype=torch.long)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for img_f, text_f, labels in train_loader:
            optim.zero_grad()
            logits = model(img_f, text_f)
            loss = loss_fn(logits, labels)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(labels)
        train_loss = total_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_img_t, val_text_t)
            val_acc = (val_logits.argmax(-1) == val_labels_t).float().mean().item()

        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "embed_dim": embed_dim,
                    "num_classes": num_classes,
                },
                MODEL_PATH,
            )

    print(f"best val_acc={best_val_acc:.4f}  saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
