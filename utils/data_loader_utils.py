import cv2
import numpy as np


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def make_square(img):
    if img.shape[0] > img.shape[1]:
        img = np.rollaxis(img, 1, 0)
    toppadlen = (img.shape[1] - img.shape[0])//2
    bottompadlen = img.shape[1] - img.shape[0] - toppadlen
    toppad = img[:5, :, :].mean(0, keepdims=True).astype(img.dtype)
    toppad = np.repeat(toppad, toppadlen, 0)
    bottompad = img[-5:, :, :].mean(0, keepdims=True).astype(img.dtype)
    bottompad = np.repeat(bottompad, bottompadlen, 0)
    return np.concatenate((toppad, img, bottompad), axis=0)
