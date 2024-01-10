import torch
import torch.nn as nn
from datasets import SVHN
from spsa import SPSA
from utils import Colors, print_color, load_checkpoint, save_checkpoint
from models import CustomCLIP
from train import train_epoch
from val import val_epoch
from configs import get_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(cfg):
    
    print_color(f'<====LOAD DATA====>', Colors.MAGENTA)
    dataset = SVHN(cfg)

    if not cfg.no_train or not cfg.no_val:
        train_loader, val_loader = dataset.get_train_val_data()
    
    model = CustomCLIP().to(device)
            
    criterion = nn.CrossEntropyLoss()
    spsa = SPSA(model, criterion)
        
    for epoch in range(1, cfg.n_epochs):

        if not cfg.no_train:
            train_epoch(epoch, spsa, train_loader, device)
            
        if not cfg.no_val:
            val_epoch(epoch, spsa, val_loader, device)
        
        if epoch % cfg.checkpoint == 0 or epoch == cfg.n_epochs -1:
            print_color(f'<====SAVE CKPT====>', Colors.MAGENTA)
            save_file_path = cfg.save_path / 'model_{}.pth'.format(epoch)
            save_checkpoint(save_file_path, epoch, spsa.model.coordinator.dec)
        
if __name__ == "__main__":
    cfg = get_configs()
    
    main(cfg)