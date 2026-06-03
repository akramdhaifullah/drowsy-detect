import torch
from sklearn.metrics import classification_report, confusion_matrix


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
