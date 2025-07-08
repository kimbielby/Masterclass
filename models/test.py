from imports import *
from Utils import *

test_psnr = []
test_ssim = []
test_loss = []

def test(test_loader, model, device):
    model.load_state_dict(torch.load("spill_model.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        psnr = 0
        ssim = 0
        loss = 0
        for inputs, gt in test_loader:
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs)
            loss_function = get_loss_function()
            loss += loss_function(output, gt).item()
            psnr += get_batch_psnr(output, gt)
            ssim += get_batch_ssim(output, gt, device=device)

        test_psnr.append(psnr)
        test_ssim.append(ssim)
        test_loss.append(loss)

        print(f"Test Loss: {loss:.3f}       Test PSNR: {psnr:.3f}       Test SSIM: {ssim:.3f}")

""" Getters for Lists """
def get_test_psnr_list():
    return test_psnr

def get_test_ssim_list():
    return test_ssim

def get_test_loss_list():
    return test_loss



