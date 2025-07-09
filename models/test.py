from imports import *
from Utils import *

test_loss = []
test_psnr = []
test_ssim = []

def test(test_loader, model, device):
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    model.load_state_dict(torch.load("spill_model.pth", weights_only=True))
    model.to(device)
    model.eval()

    loss_function = get_loss_function()
    num_batches = len(test_loader)

    with torch.no_grad():
        for inputs, gt in test_loader:
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs)

            total_loss += loss_function(output, gt).item()
            total_psnr += get_batch_psnr(output, gt)
            total_ssim += get_batch_ssim(output, gt, device=device)

    average_loss = total_loss / num_batches
    average_psnr = total_psnr / num_batches
    average_ssim = total_ssim / num_batches

    test_loss.append(average_loss)
    test_psnr.append(average_psnr)
    test_ssim.append(average_ssim)

    print(f"Test Loss: {average_loss:.3f}       Test PSNR: {average_psnr:.3f}       Test SSIM: {average_ssim:.3f}")

""" Getters for Lists """
def get_test_psnr_list():
    return test_psnr

def get_test_ssim_list():
    return test_ssim

def get_test_loss_list():
    return test_loss



