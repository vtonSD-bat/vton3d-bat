import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import cv2

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()


with torch.no_grad():
    kernelsize = 3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize // 2))
    kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]).reshape(1, 1, kernelsize, kernelsize)
    conv.weight.data = kernel  # torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()


import torch
import torch.nn.functional as F

def nearMean_map(array: torch.Tensor, mask: torch.Tensor, ksize: int = 9):
    """
    array: [H,W] or [B,1,H,W]
    mask:  [H,W] or [B,1,H,W]  (0/1)
    returns: same shape as array
    """

    # bring to [B,1,H,W]
    if array.ndim == 2:
        array4 = array[None, None]
    elif array.ndim == 4:
        array4 = array
    else:
        raise ValueError(f"array must be 2D or 4D, got {array.shape}")

    if mask.ndim == 2:
        mask4 = mask[None, None]
    elif mask.ndim == 4:
        mask4 = mask
    else:
        raise ValueError(f"mask must be 2D or 4D, got {mask.shape}")

    # make sure float
    array4 = array4.float()
    mask4 = mask4.float()

    # box filter via conv2d (no params)
    pad = ksize // 2
    weight = torch.ones((1, 1, ksize, ksize), device=array4.device, dtype=array4.dtype)

    num = F.conv2d(array4 * mask4, weight, padding=pad)
    den = F.conv2d(mask4, weight, padding=pad).clamp_min(1e-6)

    out = num / den

    # return in same ndim as input
    if array.ndim == 2:
        return out[0, 0]
    return out
