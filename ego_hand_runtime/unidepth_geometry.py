"""Pure CPU camera rays/Fourier geometry, independent of UniDepth/Torch classes."""
import numpy as np
from .geometry import unproject


def rays(camera, pixels):
    return unproject(np.asarray(pixels), camera).astype(np.float32)


def fourier(directions, frequencies):
    directions = np.asarray(directions, np.float32)
    frequencies = np.asarray(frequencies, np.float32)
    angles = directions[..., :, None]*frequencies* np.float32(np.pi)
    return np.concatenate([directions, np.sin(angles).reshape(*directions.shape[:-1], -1),
                           np.cos(angles).reshape(*directions.shape[:-1], -1)], -1)
