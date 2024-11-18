# Entry function to train function for a new neural network 

import argparse
import torch
from predict_utils import predict, load_category_names
from model_checkpoint import load_checkpoint

def main():
    ''' parse args passed by the user
    '''
    
    parser = argparse.ArgumentParser(
        description='Predict the name(flower) from an image using a trained model.',
        add_help=True
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image')
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to the model checkpoint')
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Return the top K most likely classes')
    
    parser.add_argument(
        '--category_names',
        type=str,
        default=None,
        help='Path to a JSON file mapping categoried to names')
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=4096,
        help='Number of hidden units')
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU for training if available')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load category names if provided
    category_names = load_category_names(args.category_names) if args.category_names else None

    # Load the model checkpoint
    model = load_checkpoint(args.checkpoint, device, args.hidden_units, category_names)
    model.to(device)

    # Predict
    top_prob, top_indices = predict(args.image_path, model, device, args.top_k)
    print(top_prob)
    print(top_indices)

    # Map indices to class names
    if category_names:
        class_names = [category_names[str(index)] for index in top_indices[0]]
    else:
        class_names = [str(index) for index in top_indices[0]]  # Convert indices to strings if no mapping

    # Print results
    print(f"Flower name: {class_names[0]} with a probability of {top_prob[0][0].item():.3f}")
    print("\nTop Predictions:")
    for i in range(len(top_prob[0])):  # Use size(1) to get the number of columns (top_k)
        print(f"{class_names[i]}: {top_prob[0][i].item():.3f}")
        
if __name__ == '__main__':
    main()
    
    