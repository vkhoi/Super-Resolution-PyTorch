import numpy as np

from PIL import Image
from math import log10


def _rgb2ycbcr(rgb):
    """
    Convert from RGB to YCbCr using ITU-R BT.601 conversion scheme.
    Wiki: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Input:
    - rgb: RGB image, type could be either PIL.Image or numpy array. If numpy
    array, it must be in np.uint8 type.
    Output:
    - res: image that has been converted to YCrCb, with the same type as the
    input.
    """
    if type(rgb) == Image.Image:
        # Remember that rgb is of PIL.Image type.
        is_image_type = True
        rgb = np.array(rgb)
    else:
        is_image_type = False

    if rgb.dtype != np.uint8:
        raise Exception('input must be in np.uint8 type')
        
    A = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112.0],
                  [112.0, -93.786, -18.214]], dtype=np.float32) / 255.
    offset = np.array([16., 128., 128.], dtype=np.float32)
    
    res = rgb.dot(A.T) + offset

    if rgb.dtype == np.uint8:
        # If rgb is of PIL.Image type, its numpy array version is also np.uint8,
        # hence code will also enter here.
        res = res.astype(np.uint8)

    if is_image_type:
        res = Image.fromarray(res, mode='YCbCr')
    
    return res


def _ycbcr2rgb(ycbcr):
    """
    Convert from YCbCr to RGB using ITU-R BT.601 conversion scheme.
    Wiki: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Input:
    - ycbcr: YCbCr image, type could be either PIL.Image or numpy array.
    If numpy array, it must be of type np.uint8.
    Output:
    - res: image that has been converted to RGB, with the same type as the
    input.
    """
    if type(ycbcr) == Image.Image:
        # Remember that ycbcr is of PIL.Image type.
        is_image_type = True
        ycbcr = np.array(ycbcr)
    else:
        is_image_type = False

    if ycbcr.dtype != np.uint8:
        raise Exception('input must be in np.uint8 type')
        
    A = np.linalg.inv(
            np.array([[65.481, 128.553, 24.966],
                      [-37.797, -74.203, 112.0],
                      [112.0, -93.786, -18.214]], dtype=np.float64)) * 255.
    offset = np.array([16., 128., 128.], dtype=np.float64)
    offset = A.dot(offset)
    
    res = ycbcr.astype(np.float64).dot(A.T) - offset
    res = np.clip(res, 0, 255)
    res = res.astype(np.uint8)

    if is_image_type:
        res = Image.fromarray(res, mode='RGB')
    
    return res


def compute_psnr(pred, gt, ignore_border=0):
    """Input must be np.uint8 array or RGB PIL Image.
    """
    pred = pred.squeeze()
    gt = gt.squeeze()

    if pred.ndim == 3:
        pred = np.uint8(_rgb2ycbcr(pred))[:,:,0]
    if gt.ndim == 3:
        gt = np.uint8(_rgb2ycbcr(gt))[:,:,0]

    if ignore_border > 0:
        pad = ignore_border // 2
        width, height = gt.shape[1], gt.shape[0]
        pred = pred[pad:height-pad,pad:width-pad]
        gt = gt[pad:height-pad,pad:width-pad]

    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    
    diff = (pred - gt)**2
    mse = np.mean(diff)
    psnr = 10 * log10((255.0**2) / mse)
    
    return psnr