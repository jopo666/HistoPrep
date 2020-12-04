import numpy as np
from PIL import Image


__all__ = [
    'PIL_to_array',
    'array_to_PIL',
    'mask_to_PIL',
]


def PIL_to_array(image: Image.Image) -> np.ndarray:
    """Convert Pillow image to numpy array."""
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    else:
        raise TypeError('Excpected {} not {}.'.format(Image.Image, type(image)))


def array_to_PIL(image: np.ndarray) -> Image.Image:
    """Convert numpy array to Pillow Image."""
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    else:
        raise TypeError('Excpected {} not {}.'.format(np.ndarray, type(image)))


def mask_to_PIL(mask: np.ndarray) -> Image.Image:
    """Normalize a numpy mask between 0 and 255 and convert to PIL image."""
    if isinstance(mask, np.ndarray):
        # Normalize between 0-255.
        mask = (mask/mask.max()) * 255
        return Image.fromarray(mask.astype(np.uint8))
    else:
        raise TypeError('Excpected {} not {}.'.format(np.ndarray, type(mask)))
