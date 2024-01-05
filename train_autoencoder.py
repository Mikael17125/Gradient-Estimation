import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from models_test import Autoencoder  # Assuming you have the Autoencoder class defined in models.py
from utils import compute_accuracy
import os

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 1000

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the autoencoder
model = Autoencoder().to(device)  # Use the Autoencoder class

# Loss and optimizer
criterion = nn.MSELoss()  # Use Mean Squared Error (MSE) loss for reconstruction
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the folder to save generated images
generated_images_folder = 'generated_images'
os.makedirs(generated_images_folder, exist_ok=True)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, data)
        total_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
            
        
        average_loss = total_loss / len(train_loader)
        print(f"Average MSE Loss: {average_loss:.4f}")

    # Save the trained model
    torch.save(model.encoder.state_dict(), 'encoder.pth')

    # Save a few example images
    with torch.no_grad():
        model.eval()
        example_inputs = data[:5]  # Take the first 5 images from the batch
        reconstructed_outputs = model(example_inputs)
        # Save the original and reconstructed images
        torchvision.utils.save_image(example_inputs, os.path.join(generated_images_folder, f'original_epoch_{epoch+1}.png'))
        torchvision.utils.save_image(reconstructed_outputs, os.path.join(generated_images_folder, f'reconstructed_epoch_{epoch+1}.png'))

# Test the model (evaluate on the test set)
model.eval()
total_loss = 0

with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, data)
        total_loss += loss.item()

average_loss = total_loss / len(test_loader)
print(f"Average MSE Loss on the test set: {average_loss:.4f}")
