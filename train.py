import time
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

SEED = 26
EPOCHS = 5
BATCH_SIZE = 16

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Датасет
weights = models.ResNet18_Weights.IMAGENET1K_V1
dataset = datasets.ImageFolder("data/trainval", transform=weights.transforms())

rng = random.Random(SEED)
train_idx, val_idx = [], []

for cls in range(len(dataset.classes)):
    indices = [i for i, y in enumerate(dataset.targets) if y == cls]
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx.extend(indices[:split])
    val_idx.extend(indices[split:])

train_loader = DataLoader(
    Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    Subset(dataset, val_idx), batch_size=BATCH_SIZE
)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

# 2. Модель
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 3. Обучение и валидация
Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

train_losses, val_losses = [], []
best_loss = float("inf")

for epoch in range(EPOCHS):
    start_time = time.perf_counter()
    model.eval()

    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device) # [BATCH_SIZE,3,224,224] & [BATCH_SIZE]

        optimizer.zero_grad()
        predictions = model(x) # [BATCH_SIZE, len(dataset.classes)]
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(y)

    train_loss /= len(train_idx)

    val_loss, correct = 0, 0
    with torch.inference_mode():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            val_loss += criterion(output, y).item() * len(y)
            correct += (output.argmax(1) == y).sum().item()

    val_loss /= len(val_idx)
    accuracy = correct / len(val_idx)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch + 1}: "
        f"train_loss={train_loss:.3f}, "
        f"val_loss={val_loss:.3f}, "
        f"val_accuracy={accuracy:.1%}, "
        f"time={time.perf_counter() - start_time:.1f}s"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            "model": model.state_dict(),
            "classes": dataset.classes,
        }, "checkpoints/best.pth")

# 4. График
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/loss.png")
plt.close()
