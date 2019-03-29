import numpy as np

from math import log10
from PIL import Image


def rgb2ycrcb(rgb):
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


def ycbcr2rgb(ycbcr):
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

    if ycbcr.dtype == np.uint8:
        T_scale = 255.
        offset_scale = 255.
    elif ycbcr.dtype == np.float32 or ycbcr.dtype == np.float64:
        T_scale = 255.
        offset_scale = 1.
    else:
        raise Exception('Invalid type in ycrcb2rgb')
        
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


def PSNR(pred, gt, ignore_border=0):
    """
    Compute PSNR between 2 arrays pred and gt. We will ignore the region within
    <ignore_border> pixels close to the border. The reason for ignoring this is
    that the evaluation benchmark of Super-Resolution models ignores a rim of
    6 + s pixels where is is the upscaling factor (according to NTIRE2017
    challenge).
    """
    if pred.dtype == np.uint8:
        peak_val = 255.
    else:
        peak_val = 1.

    y = pred.astype(np.float64)
    y_ref = gt.astype(np.float64)
    
    if ignore_border > 0:
        pad = ignore_border // 2
        width, height = y.shape[1], y.shape[0]
        y = y[pad:height-pad,pad:width-pad]
        y_ref = y_ref[pad:height-pad,pad:width-pad]
    
    diff = (y - y_ref)**2
    mse = np.mean(diff)
    psnr = 10 * log10(peak_val**2 / mse)
    
    return psnr