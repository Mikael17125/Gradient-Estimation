from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

class SVHN:
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        self.train_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224), antialias=True), 
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                     (0.26862954, 0.26130258, 0.27577711))])
        
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Resize((224, 224), antialias=True), 
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                (0.26862954, 0.26130258, 0.27577711))])

    def get_train_val_data(self):
        full_train_dataset = datasets.SVHN(root='./data', split='train', transform=self.train_transform, download=False)
        
         # Split the dataset into training and validation sets
        train_size = int(0.7 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Create a train dataset
        selected_train_indices = []
        for i in range(10):  # Assuming 10 classes in SVHN
            indices = [idx for idx, label in enumerate(train_dataset.dataset.labels) if label == i]
            selected_train_indices.extend(indices[:self.cfg.train_shot])

        # Create a validation dataset
        selected_val_indices = []
        for i in range(10):  # Assuming 10 classes in SVHN
            indices = [idx for idx, label in enumerate(val_dataset.dataset.labels) if label == i]
            selected_val_indices.extend(indices[:self.cfg.val_shot])

        # Create DataLoader for training and validation
        train_dataset = Subset(train_dataset.dataset, selected_train_indices)
        val_dataset = Subset(val_dataset.dataset, selected_val_indices)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        
        return train_loader, val_loader

    def get_test_data(self):
        # Create a training dataset
        full_test_dataset = datasets.SVHN(root='./data', split='test', transform=self.test_transform, download=False)
        
        # Create DataLoader for testing
        test_loader = DataLoader(dataset=full_test_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        
        return test_loader