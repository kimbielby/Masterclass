from imports import *
from Utils import *

valid_psnr = []
valid_ssim = []
valid_loss = []

def validate(model, valid_loader, num_val_batches, device):
    loss = 0
    psnr = 0
    ssim = 0

    model.eval()
    with torch.no_grad():
        for inputs, gt in valid_loader:
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs)
            loss_function = get_loss_function()
            loss += loss_function(output, gt).item()
            psnr += get_batch_psnr(model_output=output, gt=gt)
            ssim += get_batch_ssim(model_output=output, gt=gt, device=device)

        psnr /= num_val_batches
        ssim /= num_val_batches
        valid_psnr.append(psnr)
        valid_ssim.append(ssim)
        valid_loss.append(loss)

        print(f"Valid Loss: {loss:.3f}      Valid PSNR: {psnr:.3f}      Valid SSIM: {ssim:.3f}")

""" Getters for Lists """
def get_val_psnr_list():
    return valid_psnr

def get_val_ssim_list():
    return valid_ssim

def get_val_loss_list():
    return valid_loss

