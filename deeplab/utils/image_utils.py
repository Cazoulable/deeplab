

import io
import math
import numpy as np

from PIL import Image


def numpy_to_bytes(image_np, image_format):
    image = Image.fromarray(image_np)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image_format)
    return image_bytes.getvalue()


def mask_to_rle(mask):
    """
    mask: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    if np.max(mask) == 0:
        return math.nan

    dots = np.where(mask.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return ' '.join(list(map(str, run_lengths)))


def rle_to_mask(rle_string, height, width):
    if isinstance(rle_string, float) and math.isnan(rle_string):
        return np.zeros((height, width), dtype=np.uint8)

    rle_int = list(map(int, rle_string.split(' ')))
    rle_pair = np.array(rle_int).reshape(-1, 2)

    img = np.zeros(height * width, dtype=np.uint8)
    for index, length in rle_pair:
        index -= 1
        img[index: index + length] = 1

    img = img.reshape(width, height)
    img = img.T
    return img

