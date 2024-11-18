# Entry function to train function for a new neural network 

import argparse
import train_model_utils

def main():
    ''' parse args passed by the user
    '''
    
    parser = argparse.ArgumentParser(
        description='Train a neural network on a dataset.',
        add_help=True
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Path to dataset directory')
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='.',
        help='Directory to save checkpoint')
    
    parser.add_argument(
        '--arch',
        type=str,
        default='vgg13',
        help='Model architecture (e.g., vgg13, resnet18)')
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.002,
        help='Learning rate')
    
    parser.add_argument(
        '--hidden_units',
        type=int,
        default=4096,
        help='Number of hidden units')
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs')
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU for training if available')
    
    results = parser.parse_args()
    
    
    train_model_utils.train_model(results)
        
if __name__ == '__main__':
    main()
    
    