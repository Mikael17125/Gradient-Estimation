import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import torchvision.utils as vutils
from spsa import SPSA
from torch.cuda.amp import autocast
from tqdm import tqdm

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.01
batch_size = 256
epochs = 5000
o, c, a, alpha, gamma = 1.0, 0.01, 0.01, 0.4, 0.1

# Load SVHN dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224)), 
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                     (0.26862954, 0.26130258, 0.27577711))])

full_train_dataset = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
full_test_dataset = datasets.SVHN(root='./data', split='test', transform=transform, download=True)

# Select only 10 images per class for training
class_limit_train = 16
selected_train_indices = []
for i in range(10):  # Assuming 10 classes in SVHN
    indices = [idx for idx, label in enumerate(full_train_dataset.labels) if label == i]
    selected_train_indices.extend(indices[:class_limit_train])

train_dataset = Subset(full_train_dataset, selected_train_indices)

# Create a validation dataset with 4 images per class
class_limit_val = 4
selected_val_indices = []
for i in range(10):  # Assuming 10 classes in SVHN
    indices = [idx for idx, label in enumerate(full_train_dataset.labels) if label == i]
    selected_val_indices.extend(indices[:class_limit_val])

val_dataset = Subset(full_train_dataset, selected_val_indices)

# Create DataLoader for training and validation
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Loss function
criterion = nn.CrossEntropyLoss()
spsa = SPSA(device)

print("START TRAINING")

b1 = 0.9
m1 = 0

est_type = 'spsa'

# Training loop
for epoch in range(1, epochs):

    # Training phase
    spsa.model.train()
    total_loss = 0
    total_acc = 0
    for step, (images, labels) in enumerate(tqdm(train_loader)):
        with autocast():
            images = images.to(device)
            labels = labels.to(device)

            ak = a / ((epoch + o) ** alpha)
            ck = c / (epoch ** gamma)

            w = torch.nn.utils.parameters_to_vector(spsa.model.coordinator.dec.parameters())
            ghat, loss, acc = spsa.estimate(w, criterion, images, labels, ck)

            if est_type == 'spsa-gc':
                if epoch > 1: m1 = b1 * m1 + ghat
                else: m1 = ghat
                accum_ghat = ghat + b1 * m1
            elif est_type == 'spsa':
                accum_ghat = ghat
            else:
                raise ValueError

            w_new = w - ak * accum_ghat

            torch.nn.utils.vector_to_parameters(w_new, spsa.model.coordinator.dec.parameters())

            total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'epoch {epoch}, training loss {average_loss}, training acc {acc}')
    
    torch.save(spsa.model.coordinator.dec.state_dict(), 'spsa_dec.pth')
    
    # Validation phase
    spsa.model.eval()
    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():
        for val_step, (val_images, val_labels) in enumerate(tqdm(val_loader)):
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_w = torch.nn.utils.parameters_to_vector(spsa.model.coordinator.dec.parameters())
            val_ghat, val_loss, val_acc = spsa.estimate(val_w, criterion, val_images, val_labels, ck)

            total_val_loss += val_loss.item()

    average_val_loss = total_val_loss / len(val_loader)
    print(f'epoch {epoch}, validation loss {average_val_loss}, validation acc {val_acc}')
