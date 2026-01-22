import os
import json
import time
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import SimpleMLP

# ----------------------------
# Config (good optimisation knobs)
# ----------------------------
BATCH_SIZE = 64
EPOCHS = 3
LR = 0.001
HIDDEN_SIZE = 128
SEED = 42
NUM_WORKERS = 0  # set to 2 or 4 for speed experiments

# ----------------------------
# Reproducibility / determinism
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings (slower but comparable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Git metadata (best-effort)
# ----------------------------
def git_cmd(args):
    try:
        return subprocess.check_output(["git"] + args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

git_commit = git_cmd(["rev-parse", "HEAD"])
git_branch = git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
git_dirty = None
try:
    dirty = subprocess.call(["git", "diff", "--quiet"])
    git_dirty = (dirty != 0)
except Exception:
    pass

# ----------------------------
# Data
# ----------------------------
transform = transforms.ToTensor()
train_set = datasets.MNIST(".", train=True, download=True, transform=transform)
test_set = datasets.MNIST(".", train=False, download=True, transform=transform)

# Make DataLoader shuffling deterministic as well
g = torch.Generator()
g.manual_seed(SEED)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    generator=g,
    pin_memory=(device == "cuda"),
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device == "cuda"),
)

# ----------------------------
# Model
# ----------------------------
model = SimpleMLP(hidden_size=HIDDEN_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ----------------------------
# Training
# ----------------------------
start_time = time.time()
last_train_loss = None

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    last_train_loss = running_loss / max(1, n_batches)

# ----------------------------
# Evaluation
# ----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / max(1, total)
runtime_s = time.time() - start_time

# ----------------------------
# LOG OUTPUT
# ----------------------------
# Human-readable lines (nice for quick eyeballing)
print(f"Accuracy: {accuracy:.4f}")
print(f"Runtime: {runtime_s:.2f}s")
print(f"TrainLoss: {last_train_loss:.4f}")
print(f"BatchSize: {BATCH_SIZE}")
print(f"LearningRate: {LR}")
print(f"HiddenSize: {HIDDEN_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Seed: {SEED}")
print(f"Device: {device}")
if git_commit:
    print(f"GitCommit: {git_commit}")
if git_branch:
    print(f"GitBranch: {git_branch}")
if git_dirty is not None:
    print(f"GitDirty: {git_dirty}")

# Machine-parsable JSON (best for extraction)
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "accuracy": float(accuracy),
    "runtime_s": float(runtime_s),
    "train_loss": float(last_train_loss) if last_train_loss is not None else None,
    "batch_size": int(BATCH_SIZE),
    "learning_rate": float(LR),
    "hidden_size": int(HIDDEN_SIZE),
    "epochs": int(EPOCHS),
    "seed": int(SEED),
    "device": device,
    "num_workers": int(NUM_WORKERS),
    "git_commit": git_commit,
    "git_branch": git_branch,
    "git_dirty": git_dirty,
}

print("--- METRICS_JSON ---")
print(json.dumps(payload))

