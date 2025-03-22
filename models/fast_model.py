import torch
import torch.nn as nn
import models.model as model_factory
import utils.image_utils as image_utils
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_model(model_name):
    num_epochs = 10
    # Get the model
    model = model_factory.get_model(model_name, num_classes=17)

    # Create data loaders
    train_loader, val_loader = image_utils.get_data()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Lists to store training and validation accuracies
    train_accuracies = []
    val_accuracies = []

    # Training loop
    print(f"Training {model_name} model...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            running_loss += loss.item()
        
        scheduler.step()

        # Calculate training accuracy for the epoch
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation phase
        val_accuracy = model_validate(model, val_loader, device)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    # Plot training and validation accuracies
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training vs Validation Accuracy ({model_name})")
    plt.legend()
    plt.show()

    # Save the trained model
    model_save_path = f"{model_name}_best_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model


def model_validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    return val_accuracy

def test_model(model_name):
    # Load the model architecture
    model = model_factory.get_model(model_name, num_classes=17)

    # Load the saved state dictionary
    model_save_path = f"{model_name}_best_model.pth"
    model.load_state_dict(torch.load(model_save_path))

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_loader = image_utils.get_test_data()

    # Evaluate the model on the test set
    test_accuracy = model_validate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")


# Train the model
# train_model("resnet50")
train_model("vit")

# Test the model
test_model("vit")