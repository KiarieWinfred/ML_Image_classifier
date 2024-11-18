
import torch
from torch import nn, optim
from data_utils import load_data
from model_checkpoint import save_checkpoint
from torchvision import models
import argparse
import json

def train_model(args):
    """
    Train a deep learning model on a dataset and save a checkpoint.
    """
    # Load data
    dataloaders, datasets = load_data(args.data_dir)
    
    # Load category names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Check device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up architecture and load pretrained model 
        
    if args.arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        input_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(input_features, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, len(cat_to_name)),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    elif args.arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        input_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_features, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, len(cat_to_name)),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    else:
        raise ValueError(f"Unsupported architecture {args.arch}")

    
    model.to(device)
    

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_after = 20

    # Training loop
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Validation
            if steps % print_after == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.float()).item()

                # Print results
                print(f"Epoch {epoch+1}/{epochs} .. "
                      f"Train loss: {running_loss/print_after:.3f} .. "
                      f"Validation loss: {test_loss/len(dataloaders['valid']):.3f} .. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
                
    # Test the network on the test data

    model.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.float()).item()

    print(f"The Test Accuracy: {accuracy/len(dataloaders['test']):.3f}")

    # Save checkpoint
    save_checkpoint(model, datasets['train'].class_to_idx, args.arch, args.save_dir)

    
