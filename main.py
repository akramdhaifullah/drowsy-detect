import glob
import os
import sys
from collections import Counter

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from utils.dataset import MRLEyeDataset
from utils.splits import subject_wise_split


def evaluate_model(model, dataloader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=["Closed", "Open"]))

    class_names = ["Closed", "Open"]
    y_true_names = [class_names[i] for i in y_true]
    y_pred_names = [class_names[i] for i in y_pred]
    print("\nConfusion Matrix")
    print(confusion_matrix(y_true_names, y_pred_names, labels=class_names))


def main():
    print("Hello from drowsy-detect!")

    # ----- DEVICE -----

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- TRANSFORMS -----

    # Caltech101 Transforms
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #         transforms.Lambda(lambda img: img.convert("RGB")),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # mrlEyes Transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # ----- DATASET -----

    # Caltech101 Dataset
    # dataset = datasets.Caltech101(root="./data", download=True, transform=transform)

    # mrlEyes Dataset
    image_paths = glob.glob(
        os.path.join("./data/mrlEyes_2018_01", "**", "*.png"), recursive=True
    )
    train_files, test_files = subject_wise_split(image_paths, test_size=0.2)

    train_dataset = MRLEyeDataset(train_files, transform=transform)
    test_dataset = MRLEyeDataset(test_files, transform=transform)

    # ----- DATA SAMPLE INSPECTION -----

    image, label = train_dataset[0]

    print(f"type(image): {type(image)}")
    print(f"image.shape: {image.size}")
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

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable: {trainable_params:,}")
    print(f"Total: {total_params:,}")

    sys.exit(0)

    model = model.to(device)

    # ----- LOSS FUNCTION -----

    criterion = nn.CrossEntropyLoss()

    # ----- OPTIMIZER -----

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    # ----- DATALOADERS -----
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # ----- TRAINING LOOP -----

    num_epochs = 5
    for epoch in range(num_epochs):
        # Training
        model.train()

        running_loss = 0.0

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

            # Print average loss
            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)

                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

    # ----- SAVE MODEL -----

    # torch.save(model.state_dict(), "./models/mobilenet_caltech101.pth")
    torch.save(model.state_dict(), "./models/mobilenet_mrlEyes_subject-split.pth")
    print("Model saved!")

    # ----- EVALUATION -----

    evaluate_model(model, test_loader, device)

    # ----- END -----


if __name__ == "__main__":
    main()
