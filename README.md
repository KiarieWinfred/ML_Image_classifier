# My Image classfier Project

This is a deep learning-powered **Image Classifier** designed to identify different flower species. The project uses a pre-trained neural network model to classify images of flowers into their respective categories with a high degree of accuracy. 

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Acknowledgments](#acknowledgments)

## Project Overview

This project demonstrates how to build and use a neural network-based image classifier. The trained model can:
- Classify images of flowers into various species based on a provided dataset.
- Display the top K predictions with their respective probabilities.
- Map predicted indices to actual flower names for better interpretability.

The project was developed to showcase my skills in:
- Deep learning with PyTorch.
- Building and training neural networks.
- Implementing image preprocessing pipelines.
- Deploying machine learning models for inference.

## Features

- **Pre-Trained Model:** Leverages a transfer learning approach using models like VGG or ResNet.
- **Top-K Predictions:** Displays the top K predicted categories along with probabilities.
- **Category Mapping:** Converts indices to human-readable flower names.
- **GPU Support:** Allows inference on GPU for faster computation.

## Requirements

To run this project, you need the following:

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- PIL (Pillow)
- Argparse


## Examples of how to run
```
python train.py flowers --arch vgg16 --epochs 2 --gpu --save_dir ./checkpoints
```

```
python predict.py flowers/test/1/image_06743.jpg ./checkpoints/checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```
