from imports import *
from Utils import *

test_loss = []
test_psnr = []
test_ssim = []

def test(test_loader, model, device, save_an_image=False, save_path="test_output.png"):
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    saved = False

    model.load_state_dict(torch.load("spill_model.pth", weights_only=True))
    model.to(device)
    model.eval()

    loss_function = get_loss_function()
    num_batches = len(test_loader)

    with torch.no_grad():
        for idx, (inputs, gt) in enumerate(test_loader):
            inputs, gt = inputs.to(device), gt.to(device)
            output = model(inputs)

            total_loss += loss_function(output, gt).item()
            total_psnr += get_batch_psnr(output, gt)
            total_ssim += get_batch_ssim(output, gt, device=device)

            if save_an_image and not saved:
                output_np = output[0].detach().cpu().permute(1, 2, 0).numpy()
                output_np = (output_np * 255).astype(np.uint8)
                output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, output_bgr)
                print("Output stats:")
                print("Min:", output_bgr.min().item(), "Max:", output_bgr.max().item())
                print("Mean:", output_bgr.mean().item())

                print(f"Saved test output to {save_path}")
                saved = True

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



