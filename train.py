import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Autoencoder, AutoencoderCNNClassifier
import torchvision.utils as vutils
import os

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the autoencoder
model = AutoencoderCNNClassifier().to(device)  # Use the Autoencoder class

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Use Mean Squared Error (MSE) loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a directory to save generated images
os.makedirs("generated_images", exist_ok=True)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

# Test the model (evaluate on the test set)
model.eval()
total_loss = 0

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

average_loss = total_loss / len(test_loader)
print(f"Average CE Loss on the test set: {average_loss:.4f}")