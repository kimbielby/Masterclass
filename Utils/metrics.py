import torch.nn as nn
from torch.optim import Adam

loss_function = nn.MSELoss()
optimiser = Adam

def get_batch_accuracy(output, target, N):

    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / N

""" Getters for Loss and Optimisers """
def get_loss_function():
    return loss_function

def get_optimiser():
    return optimiser
