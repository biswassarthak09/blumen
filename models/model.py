import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import timm

def get_model(model_name, num_classes):
    if model_name == "resnet50":
        return get_image_net_model(num_classes)
    elif model_name == "vit":
        return get_vision_transformer_model(num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_image_net_model(num_classes):
    # Load a pre-trained ResNet50 model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # # Freeze all layers except the final layer
    # for param in model.parameters():
    #     param.requires_grad = False

    # Unfreeze the final layer
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def get_vision_transformer_model(num_classes):
    # Load a pre-trained Vision Transformer model
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes = num_classes)

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers
    # In ViT, the last few layers include:
    # 1. The classification head (model.head)
    # 2. The last few transformer blocks (model.blocks[-n:])
    num_unfrozen_blocks = 4  # Number of transformer blocks to unfreeze (adjust as needed)

    # Unfreeze the classification head
    for param in model.head.parameters():
        param.requires_grad = True

    # Unfreeze the last few transformer blocks
    for block in model.blocks[-num_unfrozen_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    return model