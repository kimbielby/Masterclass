from imports import *
from Utils import *

test_loss = []
test_psnr = []
test_ssim = []

def test(test_loader, model, device, checkpoint_path):
    """
    Standard test function. Prints average Loss, PSNR, and SSIM on test set
    :param test_loader: Dataloader with test images
    :param model: UNet model
    :param device: GPU or CPU
    :param checkpoint_path: Where the checkpoint to load lives
    """
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_spill_percent = 0.0
    total_avg_greenness = 0.0

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(device)
    model.eval()

    loss_function = get_loss_function()
    num_batches = len(test_loader)
    num_images = len(test_loader.dataset)

    with torch.no_grad():
        for inputs, gt in test_loader:
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs).clamp(min=0, max=1)

            # Metrics: Loss, PSNR, SSIM
            total_loss += loss_function(output, gt).item()
            total_psnr += get_batch_psnr(output, gt)
            total_ssim += get_batch_ssim(output, gt, device=device)

            for i in range(output.shape[0]):
                # Convert output to bgr image
                img_np = output[i].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                output_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Metrics: Spill percent, Greenness
                spill_percent, avg_greenness = check_residual_spill(img_bgr=output_bgr)
                total_spill_percent += spill_percent
                total_avg_greenness += avg_greenness

    # Averages
    average_loss = total_loss / num_batches
    average_psnr = total_psnr / num_batches
    average_ssim = total_ssim / num_batches
    average_spill_percent = total_spill_percent / num_images
    average_avg_greenness = total_avg_greenness / num_images

    test_loss.append(average_loss)
    test_psnr.append(average_psnr)
    test_ssim.append(average_ssim)

    print(f"Test Loss: {average_loss:.3f}       Test PSNR: {average_psnr:.3f}       Test SSIM: {average_ssim:.3f}")
    print(f"Residual Spill: {average_spill_percent:.2f}%    Average Greenness: {average_avg_greenness:.4f}")

""" Getters for Lists """
def get_test_psnr_list():
    return test_psnr

def get_test_ssim_list():
    return test_ssim

def get_test_loss_list():
    return test_loss



