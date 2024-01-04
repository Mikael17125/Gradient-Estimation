import torch
import torch.nn as nn

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
    
class AutoencoderCNNClassifier(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(AutoencoderCNNClassifier, self).__init__()
        self.autoencoder = Autoencoder()
        self.classifier = CNNClassifier(num_classes)
        for name, param in self.autoencoder.named_parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x = self.autoencoder(x)
        x = self.classifier(x)
        return x

    def custom_backward(self, loss):
        # Your custom backward process here
        # Example: Multiply gradients by a scaling factor
        scale_factor = 0.1
        for param in self.parameters():
            if param.requires_grad:
                param.grad.data *= scale_factor

    def backward(self, loss, retain_graph=False):
        # Call the regular backward method
        super(AutoencoderCNNClassifier, self).backward(loss, retain_graph=retain_graph)
        # Call your custom backward process
        self.custom_backward(loss)

# Instantiate the combined model
autoencoder_with_classifier = AutoencoderCNNClassifier()

# Forward pass
input_data = torch.randn(1, 1, 28, 28)  # Adjust input size accordingly
output = autoencoder_with_classifier(input_data)
breakpoint()
target = torch.randint(0, 10, (1,)).long()

# Compute cross-entropy loss
loss = torch.nn.functional.cross_entropy(output, target)
# Backward pass with custom backward process
autoencoder_with_classifier.backward(loss)

# Access the gradients and perform optimization step if needed
optimizer = torch.optim.SGD(autoencoder_with_classifier.parameters(), lr=0.01)
optimizer.step()
