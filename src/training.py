# ----- IMPORT -----

import glob
import os
from collections import Counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.transforms import InterpolationMode

from src.dataset import MRLEyeDataset
from src.evaluate import evaluate_model
from src.splits import subject_wise_split

# ----- DEVICE -----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- TRANSFORMS -----

# Training transforms (with augmentation)
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Validation/Test transforms (no augmentation)
eval_transform = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ----- DATASET -----

# mrlEyes Dataset
image_paths = glob.glob(
    os.path.join("./data/mrlEyes_2018_01", "**", "*.png"), recursive=True
)

train_files, val_files, test_files = subject_wise_split(
    image_paths,
    test_size=0.2,
    val_size=0.1,
)

train_dataset = MRLEyeDataset(train_files, transform=train_transform)
val_dataset = MRLEyeDataset(val_files, transform=eval_transform)
test_dataset = MRLEyeDataset(test_files, transform=eval_transform)

# ----- SPLIT VERIFICATION -----

print(f"Train images: {len(train_files)}")
print(f"Val images: {len(val_files)}")
print(f"Test images: {len(test_files)}")

train_subjects = {os.path.basename(os.path.dirname(p)) for p in train_files}
val_subjects = {os.path.basename(os.path.dirname(p)) for p in val_files}
test_subjects = {os.path.basename(os.path.dirname(p)) for p in test_files}

print(f"Train subjects: {len(train_subjects)}")
print(f"Val subjects: {len(val_subjects)}")
print(f"Test subjects: {len(test_subjects)}")

# ----- DATA SAMPLE INSPECTION -----

image, label = train_dataset[0]

print(f"image.shape: {image.shape}")
print(f"label: {label}")

# ----- LABEL DISTRIBUTION INSPECTION -----

train_labels = []

for path in train_files:
    filename = os.path.basename(path)
    filename = filename.replace(".png", "")

    label = int(filename.split("_")[4])

    train_labels.append(label)

print(Counter(train_labels))

# ----- MODEL -----

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze pretrained backbone
for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[-3:].parameters():
    param.requires_grad = True

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 2)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable_params:,}")
print(f"Total: {total_params:,}")

model = model.to(device)

# ----- LOSS FUNCTION -----

criterion = nn.CrossEntropyLoss()

# ----- OPTIMIZER -----

# Feature Extraction
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# Fine-Tuning
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
)

# ----- DATALOADERS -----
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ----- TRAINING LOOP -----
best_val_acc = 0.0

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

num_epochs = 5

for epoch in range(num_epochs):
    # ----- TRAIN -----
    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        preds = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        if batch_idx % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f}"
            )

    # Epoch training metrics
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct / total

    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # ----- VALIDATION -----
    model.eval()

    val_running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = correct / total

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {epoch_train_loss:.4f} "
        f"Train Acc: {epoch_train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    # ----- SAVE BEST MODEL -----

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "./models/mobilenet_mrlEyes-fine-tuning_best.pth",
        )

        print(f"New best model saved (val_acc={val_acc:.4f})")

# ----- EVALUATION -----

model.load_state_dict(
    torch.load("./models/mobilenet_mrlEyes-fine-tuning_best.pth", map_location=device)
)
model.to(device)
evaluate_model(model, test_loader, device)

# ----- PLOT METRICS -----

# Loss Curve
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_losses, label="Training Loss")

plt.plot(epochs, val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig(
    "./figures/loss_curve.png",
    bbox_inches="tight",
    dpi=300,
)
plt.close()

# Accuracy Curve
plt.figure(figsize=(8, 5))

plt.plot(epochs, train_accuracies, label="Training Accuracy")

plt.plot(epochs, val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig(
    "./figures/accuracy_curve.png",
    bbox_inches="tight",
    dpi=300,
)
plt.close()
