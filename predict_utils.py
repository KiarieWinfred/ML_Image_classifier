import torch
from torchvision import models, transforms
from model_checkpoint import load_checkpoint
import argparse
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path):
    """
    Process an image path into a PyTorch tensor with balanced resizing and adjusted cropping.
    """
    with Image.open(image_path) as pil_image:
        # Get current dimensions
        width, height = pil_image.size
        
        # Calculate dimensions to maintain aspect ratio
        target_size = 224  # Final desired size
        
        # Add some extra padding space for vertical adjustment
        padding_factor = 0.9
        
        # Determine which dimension to match
        if width > height:
            new_height = int(target_size * padding_factor)  # Make height slightly larger
            new_width = int(width * (new_height / height))
        else:
            new_width = target_size
            new_height = int(height * (target_size / width) * padding_factor)
            
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Center crop with slight downward shift
        left = (new_width - target_size) // 2
        # Adjust the top crop point to move image down
        shift_factor = 0
        top = (new_height - target_size) // 2 + int(target_size * shift_factor)
        right = left + target_size
        bottom = top + target_size
        
        # Ensure we don't crop beyond image boundaries
        if bottom > new_height:
            bottom = new_height
            top = new_height - target_size
        
        pil_image = pil_image.crop((left, top, right, bottom))
        
        # Convert to numpy array and normalize
        np_image = np.array(pil_image, dtype=np.float32) / 255
        means = np.array([0.485, 0.456, 0.406])
        deviations = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - means) / deviations
        np_image = np_image.transpose((2, 0, 1))
        
        # Convert to tensor
        tensor = torch.tensor(np_image, dtype=torch.float32) # Ensure tensor is float32
        
        return tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# def predict(image_path, model, top_k, device):
#     """
#     Predict the class (or classes) of an image using a trained deep learning model.
#     """
#     # Process the image
#     image_tensor = process_image(image_path).unsqueeze(0)  # Add batch dimension
#     image_tensor = image_tensor.to(device)
    
#     # Display the processed image
#     imshow(image_tensor.squeeze().cpu())  # Remove batch dimension for display
    
#     # Set the model to evaluation mode
#     model.eval()
    
#     with torch.no_grad():
#         outputs = model(image_tensor)
        
#         # Flatten the output if needed
#         if outputs.dim() == 4:  # Check if the output has 4 dimensions
#             outputs = torch.flatten(outputs, start_dim=1)
        
#         probabilities = torch.exp(outputs)
    
#     # Get top K probabilities and indices
#     top_prob, top_indices = probabilities.topk(top_k, dim=1)
#     return top_prob.squeeze().cpu().numpy(), top_indices.squeeze().cpu().numpy()

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path).unsqueeze(0).to(device)
    model.eval()
    
    with torch.no_grad():
        ps = torch.exp(model.forward(image))
    
    top_prob, top_indices = ps.topk(topk, dim=1)
    
    # Move results back to the CPU for further processing
    return top_prob.cpu().numpy(), top_indices.cpu().numpy()


def load_category_names(file_path):
    """
    Load a JSON file mapping category indices to class names.
    """
    with open(file_path, 'r') as f:
        return json.load(f)
