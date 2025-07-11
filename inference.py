import cv2
import numpy as np
import torch

def infer(img_path, model, device, output_path="despilled.png"):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image {img_path}")

    # Pad image
    padded_img, pad_info = pad_image(img)

    # Convert to RGB
    rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Convert to tensor
    img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Infer
    with torch.no_grad():
        output = model(img_tensor).clamp(0, 1)

    # Convert output to image
    output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output_np = (output_np * 255).astype(np.uint8)

    # Convert to BGR
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    # Remove padding
    unpadded_img = unpad_image(output_bgr, pad_info)

    # Save image
    cv2.imwrite(output_path, unpadded_img)
    print(f"Saved image to {output_path}")

def pad_image(img, pad_colour=255):
    height, width = img.shape[:2]
    diff = abs(height - width)
    if height > width:
        left = diff // 2
        right = diff - left
        top, bottom = 0, 0
    else:
        top = diff // 2
        bottom = diff - top
        left, right = 0, 0
    pad_info = (top, bottom, left, right)
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=pad_colour)
    return padded_img, pad_info

def unpad_image(padded_img, pad_info):
    top, bottom, left, right = pad_info
    height, width = padded_img.shape[:2]
    return padded_img[top:height-bottom, left:width-right]

