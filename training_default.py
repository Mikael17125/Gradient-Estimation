import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from spsa import SPSA
from torch.cuda.amp import autocast
from utils import Colors, print_color
import time

from tqdm import tqdm
# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.01
batch_size = 64
epochs = 5000
o, c, a, alpha, gamma = 1.0, 0.01, 0.01, 0.4, 0.1

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

# Loss function
criterion = nn.CrossEntropyLoss()
spsa = SPSA(device)

print_color("START TRAINING", Colors.MAGENTA)

b1 = 0.9
m1 = 0

est_type = 'spsa'

no_train = True

if not(no_train):
    # Training loop
    for epoch in range(1, epochs):

        # Training phase
        total_loss = 0
        total_acc = 0
        
        start_time = time.time()
        for step, (images, labels) in enumerate((train_loader)):

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
        end_time = time.time()
        
        print_color(f'epoch {epoch:4d} | training loss {average_loss:5f} | training acc {acc:5f} | Elapsed Time {end_time - start_time:.3f}', Colors.RED)
        
        
        torch.save(spsa.model.coordinator.dec.state_dict(), f'spsa_dec_2.pth')
        
        # Validation phase
        total_val_loss = 0
        total_val_acc = 0
        start_time = time.time()
        with torch.no_grad():
            for val_step, (val_images, val_labels) in enumerate((val_loader)):
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_w = torch.nn.utils.parameters_to_vector(spsa.model.coordinator.dec.parameters())
                val_ghat, val_loss, val_acc = spsa.estimate(val_w, criterion, val_images, val_labels, ck)
        
        end_time = time.time()
        print_color(f'epoch {epoch:4d} | validation loss {val_loss:5f} | validation acc {val_acc:5f} | Elapsed Time {end_time - start_time:.3f}\n', Colors.GREEN)
        
print_color(f'EVALUATING', Colors.MAGENTA)

spsa.model.coordinator.dec.load_state_dict(torch.load("/home/mikael/Code/custom_backward/spsa_dec.pth"))

spsa.model.eval()
total_test_loss = 0
total_test_acc = 0
correct = 0

with torch.no_grad():
    for test_step, (test_images, test_labels) in enumerate(tqdm(test_loader)):
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        outputs = spsa.model(test_images)
        test_loss = criterion(outputs, test_labels)

        total_test_loss += test_loss.item()

        test_prediction = torch.argmax(outputs, dim=1)
        correct += (test_labels == test_prediction).float().sum()

accuracy = 100 * correct / len(full_test_dataset)        
average_test_loss = total_test_loss / len(test_loader)

print(f'Test loss {average_test_loss} | Accuracy {accuracy}\n')
