'''
Date: 2023-11-14
Authors: Kuniaki Iwanami, Masa Someha, Juan Pablo Triana Martinez
Project: CS236 Final Project, VAE for X-rays images.
'''

from torch.nn import functional as F
from torchvision import datasets, transforms

import torch

if __name__ == "main":
    # Check if cuda device is in
    device = torch.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    #train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)