import matplotlib.pyplot as plt
import numpy as np
from models import train, validate, test

def plot_results():
    # Get metrics
    train_psnr_list = train.get_train_psnr_list()
    train_ssim_list = train.get_train_ssim_list()

    valid_psnr_list = validate.get_val_psnr_list()
    valid_ssim_list = validate.get_val_ssim_list()

    # PSNR
    plt.figure(figsize=(5, 2))
    plt.plot(train_psnr_list, label='Train')
    plt.plot(valid_psnr_list, label='Valid')
    plt.xlabel('Epoch')
    plt.title('PSNR')
    plt.legend()
    plt.savefig("psnr_plot.png", bbox_inches='tight')
    plt.show()
    # SSIM
    plt.figure(figsize=(5, 2))
    plt.plot(train_ssim_list, label='Train')
    plt.plot(valid_ssim_list, label='Valid')
    plt.xlabel('Epoch')
    plt.title('SSIM')
    plt.legend()
    plt.savefig("ssim_plot.png", bbox_inches='tight')
    plt.show()


""" For Segment Anything """

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)





