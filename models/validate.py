from imports import *
from Utils import *

valid_accuracy = []
valid_loss = []

def validate(model, valid_loader, num_samples_batch, device):
    accuracy = 0
    loss = 0

    model.eval()
    with torch.no_grad():
        for inputs, gt in valid_loader:
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs)

    return


