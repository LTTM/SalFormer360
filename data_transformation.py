import random
import torch
import torchvision.transforms.functional as TF

SIZE_W = 384
SIZE_H = 224

def normalize_tensor_max(tensor):
    return tensor / (tensor.max())

def image_transform(img1, img2, target, bias, dataset_name):
    # Ensure both images are PIL Images or NumPy ndarrays before applying transformations
    if isinstance(img1, torch.Tensor) or isinstance(img2, torch.Tensor):
        raise TypeError(
            "Inputs should be PIL Images or NumPy ndarrays before transformation."
        )

    # Apply the same transformation to both images
    img1 = TF.resize(img1, [SIZE_H, SIZE_W])
    img2 = TF.resize(img2, [SIZE_H, SIZE_W])

    if dataset_name != "360":
        target = target.unsqueeze(0)

    #target = target.unsqueeze(0)  # shape becomes [1, 256, 512]
    target = TF.resize(target, [SIZE_H, SIZE_W])
    bias = TF.resize(bias, [SIZE_H, SIZE_W])

    img1 = TF.to_tensor(img1)
    img2 = TF.to_tensor(img2)
    bias = TF.to_tensor(bias)

    # Random horizontal flip
    if random.random() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
        target = TF.hflip(target)
        bias = TF.hflip(bias)

    # Random vertical flip
    if random.random() > 0.95:
        img1 = TF.vflip(img1)
        img2 = TF.vflip(img2)
        target = TF.vflip(target)
        bias = TF.vflip(bias)

    # Apply additional transformations as needed
    # For example, color jitter
    # Random brightness adjustment
    brightness_factor = random.uniform(0.7, 1.3)
    img1 = TF.adjust_brightness(img1, brightness_factor)
    img2 = TF.adjust_brightness(img2, brightness_factor)

    # Random contrast adjustment
    contrast_factor = random.uniform(0.7, 1.3)
    img1 = TF.adjust_contrast(img1, contrast_factor)
    img2 = TF.adjust_contrast(img2, contrast_factor)

    # Random saturation adjustment
    saturation_factor = random.uniform(0.7, 1.3)
    img1 = TF.adjust_saturation(img1, saturation_factor)
    img2 = TF.adjust_saturation(img2, saturation_factor)

    bias = normalize_tensor_max(bias)

    return img1, img2, target, bias


def test_transform(img1, img2, target, bias, dataset_name):
    # Ensure both images are PIL Images or NumPy ndarrays before applying transformations
    if isinstance(img1, torch.Tensor) or isinstance(img2, torch.Tensor):
        raise TypeError(
            "Inputs should be PIL Images or NumPy ndarrays before transformation."
        )

    # Apply the same transformation to both images
    img1 = TF.resize(img1, [SIZE_H, SIZE_W])
    img2 = TF.resize(img2, [SIZE_H, SIZE_W])
    bias = TF.resize(bias, [SIZE_H, SIZE_W])

    if dataset_name != "360":
        target = target.unsqueeze(0)
    
    #target = target.unsqueeze(0)  # shape becomes [1, 256, 512]
    target = TF.resize(target, [SIZE_H, SIZE_W])

    img1 = TF.to_tensor(img1)
    img2 = TF.to_tensor(img2)
    bias = TF.to_tensor(bias)

    bias = normalize_tensor_max(bias)
  
    return img1, img2, target, bias
