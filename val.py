import torch
from utils import print_color, Colors
import time

def val_epoch(epoch, data_loader, spsa, device, tb_writer):
    start_time = time.time()
    with torch.no_grad():
        for _, (images, labels) in enumerate((data_loader)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = spsa.model(images)
            loss = spsa.criterion(outputs, labels)

            correct += (labels == torch.argmax(outputs, dim=1)).float().sum()

    end_time = time.time()

    print_color(f'Validating {epoch:2d} | Elapsed Time {end_time - start_time:.3f} | Loss {loss:.5f}', Colors.YELLOW)
