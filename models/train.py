from Utils import *

train_psnr = []
train_ssim = []
train_loss = []

def train(model, train_loader, num_samples_batch, device, lr=1e-4):
    loss = 0.0
    psnr = 0.0
    ssim = 0.0

    model.train()
    for inputs, gt in train_loader:
        inputs, gt = inputs.to(device), gt.to(device)

        output = model(inputs)

        optimiser = get_optimiser()
        optimiser = optimiser(model.parameters(), lr=lr)
        optimiser.zero_grad()

        loss_function = get_loss_function()
        batch_loss = loss_function(output, gt)
        batch_loss.backward()

        optimiser.step()

        loss += batch_loss.item()
        psnr += get_batch_psnr(model_output=output, gt=gt)
        ssim += get_batch_ssim(model_output=output, gt=gt, device=device)

    psnr /= num_samples_batch
    ssim /= num_samples_batch
    train_psnr.append(psnr)
    train_ssim.append(ssim)
    train_loss.append(loss)
    print(f"Train Loss: {loss:.3f}, Train PSNR: {psnr:.3f}, Train SSIM: {ssim:.3f}")

""" Getters for Lists """
def get_train_psnr_list():
    return train_psnr

def get_train_ssim_list():
    return train_ssim

def get_train_loss_list():
    return train_loss
