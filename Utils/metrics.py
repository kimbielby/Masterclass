from imports import *

loss_function = nn.MSELoss()
optimiser = Adam

def get_batch_accuracy(output, target, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / N

def get_batch_psnr(model_output, gt):
    metric = PeakSignalNoiseRatio()
    metric.update(input=model_output, target=gt)
    return metric.compute()

def get_batch_ssim(model_output, gt, device):
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return metric(preds=model_output, target=gt)

def get_ave_residual_spill():

    return

""" Getters for Loss and Optimisers """
def get_loss_function():
    return loss_function

def get_optimiser():
    return optimiser
