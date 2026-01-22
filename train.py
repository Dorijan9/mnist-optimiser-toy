import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleMLP

# Config (these are great targets for optimisation)
BATCH_SIZE = 64
EPOCHS = 3
LR = 0.001
HIDDEN_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
transform = transforms.ToTensor()
train_set = datasets.MNIST(".", train=True, download=True, transform=transform)
test_set = datasets.MNIST(".", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

# Model
model = SimpleMLP(hidden_size=HIDDEN_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# Training
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total
runtime = time.time() - start_time

# LOG OUTPUT (IMPORTANT)
print(f"Accuracy: {accuracy:.4f}")
print(f"Runtime: {runtime:.2f}s")
print(f"BatchSize: {BATCH_SIZE}")
print(f"LearningRate: {LR}")
print(f"HiddenSize: {HIDDEN_SIZE}")
