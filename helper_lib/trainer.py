import torch


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=1,
):
    """
    Train a PyTorch model.

    model.train() is important because it puts the model into training mode.
    """

    model.to(device)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total

                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy:.2f}%"
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total

        print(
            f"Epoch [{epoch + 1}/{epochs}] finished - "
            f"Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )

    return model