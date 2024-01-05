import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

train_dataset = datasets.SVHN(root='./data', split='train', transform=transform, download=True)
test_dataset = datasets.SVHN(root='./data', split='test', transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Loss function
criterion = nn.CrossEntropyLoss()
spsa = SPSA(device)

print("START TRAINING")

b1 = 0.9
m1 = 0

est_type = 'spsa'

# Training loop
for epoch in range(1, epochs):
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
    print(f'epoch {epoch}, loss {average_loss}')
    
     # Save autoencoder output images every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            spsa.model.eval()
            output_images = spsa.model.autoencoder(images)
            vutils.save_image(output_images, f"generated_images/epoch_{epoch}_output.png", normalize=True)
