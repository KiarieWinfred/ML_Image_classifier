from torchvision import datasets, transforms
import torch

def load_data(data_dir):
    """
    Loads the datasets and applies transformations for training, validation, and testing.
    Args:
        data_dir: Directory containing the datasets (train, valid, test).
    Returns:
        dataloaders: Dictionary containing data loaders for train, validation, and test sets.
        datasets: Dictionary containing datasets for train, validation, and test sets.
    """
    # Define transforms for training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train']),
        'valid': datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid']),
        'test': datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test'])
    }

    # Define dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=64, shuffle=True, num_workers=4
        ),
        'valid': torch.utils.data.DataLoader(
            image_datasets['valid'], batch_size=64, shuffle=False, num_workers=4
        ),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'], batch_size=64, shuffle=False, num_workers=4
        )
    }

    return dataloaders, image_datasets
