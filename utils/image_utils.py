from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
from torch.utils.data import DataLoader
import src.data_augmentations as data_augment
from torchvision import datasets


def get_data():
    # Load training dataset with augmentation
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=data_augment.train_transform)

    # Load validation dataset without augmentation
    val_dataset = datasets.ImageFolder(root='dataset/val', transform=data_augment.val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

def get_test_data():
    # Load test dataset without augmentation
    test_dataset = datasets.ImageFolder(root='dataset/test', transform=data_augment.val_transform)

    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_loader


def count_images(dataset):
# Load dataset
    dataset = ImageFolder(root="dataset/train")

    # Count occurrences of each class
    class_counts = {dataset.classes[i]: sum(1 for label in dataset.targets if label == i) for i in range(len(dataset.classes))}

    # Print counts
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")


def show_augmented_images():

    train_dataset = datasets.ImageFolder(root='dataset/train', transform=data_augment.train_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Get a batch of training data
    images, labels = next(iter(train_loader))

    # Denormalize the images for visualization
    def denormalize(image):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return image * std + mean

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Training Images with Augmentation")

    # Create grid and permute dimensions for correct visualization
    grid_image = vutils.make_grid(denormalize(images[:8]))  # Create a grid from first 8 images
    grid_image = grid_image.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C) for matplotlib

    plt.imshow(grid_image)  # Display the image grid
    plt.show()


# show_augmented_images()
# count_images("dataset/train")