import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Autoencoder, BlackVIP
import torchvision.utils as vutils
import os
from spsa import SPSA
from torch.cuda.amp import autocast


# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
epochs = 1000

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the BLACK VIP
model = BlackVIP().to(device)
model.classifier.load_state_dict(torch.load("/home/mikael/Code/custom_backward/classifier.pth"))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Use Mean Squared Error (MSE) loss for reconstruction
spsa = SPSA()

# Training loop
for epoch in range(epochs):
    model.train()
    
    total_loss = 0
    for step, (images, labels) in enumerate(train_loader):
        with autocast():
            images = images.to(device)
            labels = labels.to(device)
            
            w = torch.nn.utils.parameters_to_vector(model.autoencoder.parameters())
                    
            loss = spsa.estimate(epoch, model, criterion, images, labels)
            total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'EPOCH {epoch}, LOSS {average_loss}')