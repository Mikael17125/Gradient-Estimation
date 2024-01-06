import torch
from utils import print_color, Colors
import time
from tqdm import tqdm

def inference(data_loader, spsa, device, tb_writer):
    with torch.no_grad():
        for _, (images, labels) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = spsa.model(images)
            loss = spsa.criterion(outputs, labels)

            correct += (labels == torch.argmax(outputs, dim=1)).float().sum()

    end_time = time.time()

    print_color(f'Accuracy {loss:.5f} | Loss {loss:.5f}', Colors.YELLOW)
