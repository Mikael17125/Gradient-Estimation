import torch
import torch.nn as nn
from datasets import SVHN
from spsa import SPSA
from utils import Colors, print_color, load_checkpoint
from models import CustomCLIP
from train import train_epoch
from val import val_epoch
from inference import inference
from configs import get_configs

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(cfg):
    
    print_color(f'<====LOAD DATA====>', Colors.MAGENTA)
    dataset = SVHN(cfg)

    if not cfg.no_train or not cfg.no_val:
        train_loader, val_loader = dataset.get_train_val_data()
        
    if not cfg.no_inference:
        test_loader = dataset.get_test_data()
    
    model = CustomCLIP().to(device)
    
    if cfg.ckpt_path:
        checkpoint = load_checkpoint(cfg.ckpt_path)
        model.coordinator.dec.load_state_dict(checkpoint["state_dict"])
        
    criterion = nn.CrossEntropyLoss()
    spsa = SPSA(model, criterion)
        
    for epoch in range(1, cfg.n_epochs):

        if not cfg.no_train:
            train_epoch(epoch, spsa, train_loader, device)
            
        if not cfg.no_val:
            val_epoch(epoch, spsa, val_loader, device)
        
        if epoch % cfg.checkpoint == 0:
            if not cfg.no_inference:
                inference(spsa, test_loader, device)
        
if __name__ == "__main__":
    cfg = get_configs()
    
    main(cfg)