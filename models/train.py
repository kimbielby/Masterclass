from Utils import *
import torch

train_loss = []
train_psnr = []
train_ssim = []

def train(model, train_loader, device, lr=1e-4):
    """
    Standard Train function. Prints average Loss, PSNR, and SSIM
    :param model: UNet model
    :param train_loader: Dataloader with training data
    :param device: GPU or CPU
    :param lr: Learning rate
    """
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    model.train()
    loss_function = get_loss_function()
    optimiser = get_optimiser()(model.parameters(), lr=lr)

    num_batches = len(train_loader)

    for inputs, gt in train_loader:
        inputs, gt = inputs.to(device), gt.to(device)

        optimiser.zero_grad()

        output = model(inputs)
        batch_loss = loss_function(output, gt)
        batch_loss.backward()
        optimiser.step()

        total_loss += batch_loss.item()
        with torch.no_grad():
            total_psnr += get_batch_psnr(model_output=output, gt=gt)
            total_ssim += get_batch_ssim(model_output=output, gt=gt, device=device)

    average_loss = total_loss / num_batches
    average_psnr = total_psnr / num_batches
    average_ssim = total_ssim / num_batches

    train_loss.append(average_loss)
    train_psnr.append(average_psnr)
    train_ssim.append(average_ssim)

    print(f"Train Loss: {average_loss:.3f}      Train PSNR: {average_psnr:.3f}         Train SSIM: {average_ssim:.3f}")


""" Getters for Lists """
def get_train_psnr_list():
    return train_psnr

def get_train_ssim_list():
    return train_ssim

def get_train_loss_list():
    return train_loss
