# %%
import cv2
import numpy as np


def save_image(array: np.array, fname: str, dtype="uint8"):
    image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR).astype(dtype)
    cv2.imwrite(
        fname,
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )
