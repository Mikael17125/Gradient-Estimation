import torch
import torch.nn as nn
from utils import Colors, print_color, load_checkpoint
from tqdm import tqdm
from models import CustomCLIP
from datasets import SVHN
from spsa import SPSA
from configs import get_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(spsa, data_loader, device):
    correct = 0
    len_data = 0
    
    with torch.no_grad():
        for _, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = spsa.model(images)
            loss = spsa.criterion(outputs, labels)

            correct += (labels == torch.argmax(outputs, dim=1)).float().sum()
            len_data += outputs.shape[0]
                
    print_color(f'Accuracy {correct/len_data:.5f} | Loss {loss:.5f}', Colors.CYAN)


def main(cfg):
    
    print_color(f'<====LOAD DATA====>', Colors.MAGENTA)
    dataset = SVHN(cfg)

    test_loader = dataset.get_test_data()
    
    model = CustomCLIP().to(device)
    
    print_color(f'<====LOAD CKPT====>', Colors.MAGENTA)
    if cfg.ckpt_path:
        checkpoint = load_checkpoint(cfg.ckpt_path)
        model.coordinator.dec.load_state_dict(checkpoint["state_dict"])
        
    criterion = nn.CrossEntropyLoss()
    spsa = SPSA(model, criterion)
        
    model = CustomCLIP().to(device)

    criterion = nn.CrossEntropyLoss()
    spsa = SPSA(model, criterion)

    inference(spsa, test_loader, device)

if __name__ == "__main__":
    cfg = get_configs()
    
    main(cfg)