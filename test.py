import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Загрузка модели
checkpoint = torch.load(
    "checkpoints/best.pth",
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

# 2. Тестовая выборка
weights = models.ResNet18_Weights.IMAGENET1K_V1
dataset = datasets.ImageFolder("data/test", transform=weights.transforms())

assert dataset.classes == classes
assert len(dataset) >= 1 # 50

loader = DataLoader(dataset, batch_size=16)

# 3. Проверка
correct = 0
index = 0

with torch.inference_mode():
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        probabilities = model(x).softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)

        for i in range(len(y)):
            path, _ = dataset.samples[index + i]

            actual = classes[y[i].item()]
            predicted = classes[predictions[i].item()]
            probability = confidence[i].item()

            correct += actual == predicted

            print(
                f"{path}: "
                f"Actual={actual}, "
                f"Predicted={predicted}, "
                f"Confidence={probability:.1%}"
            )

        index += len(y)

print(f"\nTest accuracy: {correct / len(dataset):.2%}")
print(f"Correct: {correct}/{len(dataset)}")
