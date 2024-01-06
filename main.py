import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from spsa import SPSA
from utils import Colors, print_color
from models import CustomCLIP

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
epochs = 5000

# Load SVHN dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224)), 
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                     (0.26862954, 0.26130258, 0.27577711))])

full_train_dataset = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
full_test_dataset = datasets.SVHN(root='./data', split='test', transform=transform, download=True)

# Split the dataset into training and validation sets
train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Select only 16 images per class for training
class_limit_train = 16
selected_train_indices = []
for i in range(10):  # Assuming 10 classes in SVHN
    indices = [idx for idx, label in enumerate(train_dataset.dataset.labels) if label == i]
    selected_train_indices.extend(indices[:class_limit_train])

# Create a validation dataset with 4 images per class
class_limit_val = 4
selected_val_indices = []
for i in range(10):  # Assuming 10 classes in SVHN
    indices = [idx for idx, label in enumerate(val_dataset.dataset.labels) if label == i]
    selected_val_indices.extend(indices[:class_limit_val])

# Create DataLoader for training and validation
train_dataset = Subset(train_dataset.dataset, selected_train_indices)
val_dataset = Subset(val_dataset.dataset, selected_val_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(dataset=full_test_dataset, batch_size=batch_size, shuffle=False)

model = CustomCLIP()
criterion = nn.CrossEntropyLoss()

spsa = SPSA(model, criterion)