import torch


def evaluate_model(
    model,
    data_loader,
    criterion,
    device="cpu",
):
    """
    Evaluate a PyTorch model on validation or test data.

    model.eval() is important because it puts the model into evaluation mode.
    torch.no_grad() avoids tracking gradients during evaluation.
    """

    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy