import subprocess
from PIL import Image
import numpy as np
import time

def capture():
    subprocess.call(["fswebcam", "--no-banner", "image.jpg"])
    im = Image.open("image.jpg", "r").convert('L')
    w, h = im.size
    im = im.crop((40, 70, w, h))
    im = im.resize((30, 30), Image.ANTIALIAS)
    im.save("grayscale.jpg")
    im = np.array(im)
    im = im.reshape(1, im.shape[0] * im.shape[1])
    im = (im / 255.)
    im = np.round(im, 3)
    return im

#capture()