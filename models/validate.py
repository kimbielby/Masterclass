from imports import *
from Utils import *

valid_loss = []
valid_psnr = []
valid_ssim = []

def validate(model, valid_loader, device):
    """
    Standard Validation function. Prints average Loss, PSNR, and SSIM
    :param model: UNet model
    :param valid_loader: Dataloader for validation
    :param device: GPU or CPU
    """
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    model.eval()
    loss_function = get_loss_function()

    num_batches = len(valid_loader)

    with torch.no_grad():
        for inputs, gt in valid_loader:
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs)

            total_loss += loss_function(output, gt).item()
            total_psnr += get_batch_psnr(model_output=output, gt=gt)
            total_ssim += get_batch_ssim(model_output=output, gt=gt, device=device)

    average_loss = total_loss / num_batches
    average_psnr = total_psnr / num_batches
    average_ssim = total_ssim / num_batches

    valid_loss.append(average_loss)
    valid_psnr.append(average_psnr)
    valid_ssim.append(average_ssim)

    print(f"Valid Loss: {average_loss:.3f}      Valid PSNR: {average_psnr:.3f}      Valid SSIM: {average_ssim:.3f}")

""" Getters for Lists """
def get_val_psnr_list():
    return valid_psnr

def get_val_ssim_list():
    return valid_ssim

def get_val_loss_list():
    return valid_loss

