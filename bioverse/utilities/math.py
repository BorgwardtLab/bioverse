import numpy as np


def normalize(v, axis=-1, eps=1e-8):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + eps)
