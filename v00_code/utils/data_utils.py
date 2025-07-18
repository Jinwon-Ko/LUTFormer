import cv2
import torch
import torch.nn.functional as F
import numpy as np


def clipping(x, MIN=-1.0, MAX=1.0):
    return x.clamp(min=MIN, max=MAX)

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def inverse_tanh(x, eps=1e-3):
    x = x.clamp(min=-1, max=1)
    # torch.arctanh(x)
    x1 = (x + 1).clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return 0.5 * torch.log(x1/x2)
