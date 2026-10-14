"""
utils.py  --  PERSON 1: image <-> matrix pipeline (everyone imports this file)

Conventions (do not change without telling the team):
  * An image is a float64 NumPy array of shape (H, W, 3), values in [0, 1].
  * A filter is a 3x3 matrix A that acts on a COLUMN vector [R, G, B]^T:
        new_pixel = A @ old_pixel
    so "apply A first, then B" is the single matrix  B @ A.
"""
import os
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# Core pipeline
# ----------------------------------------------------------------------
def load_image(path):
    """Read an image file -> float array (H, W, 3) in [0, 1].

    Concept: a colour image IS a matrix of numbers. Each pixel holds an
    RGB vector; we scale 0-255 down to 0-1 so we can do real arithmetic.
    """
    img = Image.open(path).convert("RGB")           # force 3 channels
    return np.asarray(img, dtype=np.float64) / 255.0


def apply_filter(img, A, clip=True):
    """Apply a 3x3 colour filter A to every pixel of img.

    Concept: stack all H*W pixels as columns of one 3 x (H*W) matrix P.
    Then ONE matrix product  A @ P  filters every pixel at once, and we
    reshape the result back into an image.

    clip=True  -> clamp to [0, 1] so the result is a valid image
    clip=False -> keep raw values (needed to test exact inverses)
    """
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (3, 3):
        raise ValueError(f"Filter must be 3x3, got {A.shape}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3), got {img.shape}")

    h, w, _ = img.shape
    pixels = img.reshape(-1, 3).T          # 3 x (H*W): one column per pixel
    new_pixels = A @ pixels                # the linear transformation
    out = new_pixels.T.reshape(h, w, 3)    # back to image shape
    return np.clip(out, 0.0, 1.0) if clip else out


def save_image(img, path):
    """Save a float (H, W, 3) image in [0, 1] as an 8-bit file. Creates folders."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    arr = np.rint(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ----------------------------------------------------------------------
# Shared helpers (used by Persons 2, 3, 4)
# ----------------------------------------------------------------------
def error_between(a, b):
    """Return (MSE, max absolute error) between two images."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d ** 2)), float(np.max(np.abs(d)))


def image_info(img):
    """Short text summary of an image array."""
    h, w, c = img.shape
    return (f"shape = {h} x {w} x {c}  ({h * w:,} pixels), "
            f"dtype = {img.dtype}, range = [{img.min():.3f}, {img.max():.3f}]")


def banner(stage, concept, purpose):
    """Print a stage header in the Concept -> Purpose -> Outcome viva format."""
    line = "=" * 72
    print(f"\n{line}\n{stage}\n  Concept : {concept}\n  Purpose : {purpose}\n{line}")


def make_sample_image(path, h=240, w=360):
    """Create a colourful synthetic test image so the code runs anywhere.
    Replace with a REAL photo (images/photo.jpg) for the final submission."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x / w, y / h, 1 - (x + y) / (w + h)], axis=-1)
    for cx, cy, r, col in [(90, 90, 50, (0.95, 0.75, 0.1)),
                           (250, 80, 40, (0.8, 0.1, 0.2)),
                           (180, 170, 55, (0.1, 0.6, 0.3))]:
        mask = (x - cx) ** 2 + (y - cy) ** 2 < r ** 2
        img[mask] = col
    save_image(img, path)
    return path


def show_side_by_side(images, titles, save_to=None, show=False):
    """Save (save_to) and/or display (show=True) several images in one row.

    show defaults to False so scripts never stop at a pop-up window;
    the Colab notebook passes show=True to display inline."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    axes = np.atleast_1d(axes)
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        fig.savefig(save_to, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
