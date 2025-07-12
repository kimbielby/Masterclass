from imports import *

loss_function = nn.MSELoss()
optimiser = Adam

def get_batch_accuracy(output, target, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / N

def get_batch_psnr(model_output, gt, data_range=1.0):
    metric = PeakSignalNoiseRatio(data_range=data_range)
    metric.update(input=model_output, target=gt)
    return metric.compute().item()

def get_batch_ssim(model_output, gt, device):
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return metric(preds=model_output, target=gt).item()

def check_residual_spill(img_bgr, threshold=0.1):
    # Normalise
    bgr = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(bgr)

    # Calculate greenness
    greenness = g - 0.5 * (r + b)

    # Create green pixel mask
    spill_mask = greenness > threshold

    total_pixels = greenness.size
    spill_pixels = np.sum(spill_mask)

    percent_spill = 100 * spill_pixels / total_pixels
    avg_greenness = np.mean(greenness[spill_mask]) if spill_pixels > 0 else 0.0

    return percent_spill, avg_greenness

""" Getters for Loss and Optimisers """
def get_loss_function():
    return loss_function

def get_optimiser():
    return optimiser
