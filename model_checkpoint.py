# Save and load checkpoint

import torch
from torch import nn
import os
from torchvision import models
from torchvision.models import VGG16_Weights



def save_checkpoint(model, class_to_idx, arch, save_dir):
    """
    Saves the model checkpoint.
    Args:
        model: Trained model
        class_to_idx: Class to index mapping
        arch: Model architecture
        save_dir: Directory to save the checkpoint
    """
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 
    checkpoint = {
        'arch': arch,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }
    checkpoint_path = f"{save_dir}/checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

def load_checkpoint(filepath, device, hidden_units, cat_to_name):
    """
    Loads a model checkpoint.
    Args:
        filepath: Path to the checkpoint file
        device: Device to load the model onto (CPU or GPU)
    Returns:
        model: Loaded model
        class_to_idx: Class to index mapping
    """
    from predict_utils import load_category_names
    checkpoint = torch.load(filepath, map_location=device)
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        input_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, len(cat_to_name)),
            nn.LogSoftmax(dim=1)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model
 
    elif arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        input_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, len(cat_to_name)),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError(f"Unsupported architecture {arch}")

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    print(f"Model loaded from {filepath}")
    return model, checkpoint['class_to_idx']
