import torch
from utils import print_color, Colors
from tqdm import tqdm

def inference(spsa, data_loader, device):
    correct = 0
    len_data = 0
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = spsa.model(images)
            loss = spsa.criterion(outputs, labels)

            correct += (labels == torch.argmax(outputs, dim=1)).float().sum()
            len_data += outputs.shape[0]
            if idx == 203:
                import pdb; pdb.set_trace()
                
    print_color(f'Accuracy {correct/len_data:.5f} | Loss {loss:.5f}', Colors.CYAN)
