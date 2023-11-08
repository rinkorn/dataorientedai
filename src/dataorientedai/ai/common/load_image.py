import numpy as np
from PIL import Image


def load_image(fname):
    image = Image.open(fname)
    image = np.asarray(image)
    # image = plt.imread('image.png')
    # image = cv2.imread(fname)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
