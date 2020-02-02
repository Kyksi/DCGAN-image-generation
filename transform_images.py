import numpy as np


# strip transparency dimension (because RGB channels are crazy in transparencent spaces)
def strip_transparency(image):
    px_transparent = image[:, :, 3] < 0.1
    image[px_transparent, 0:3] = 1
    image = image[:, :, 0:3]
    return image


def scale(x, scale_min_val=-1, scale_max_val=1):
    # scale to feature_range
    x = x * (scale_max_val - scale_min_val) + scale_min_val
    return x


def unscale(x, scale_min_val=-1, scale_max_val=1):
    # scale to (0, 1)
    x = (x - scale_min_val) / (scale_max_val - scale_min_val)
    return x


def get_batches(images, batch_size, shuffle=True):
    images = np.array(images)
    if shuffle:
        idx = np.arange(len(images))
        np.random.shuffle(idx)
        images = images[idx]
    n_batches = len(images) // batch_size
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size]
        yield x
