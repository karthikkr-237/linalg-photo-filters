"""
main.py  --  PERSON 1: runs the whole project and saves everything to output/

Usage:   python main.py                       (uses images/photo.jpg)
         python main.py --image my_pic.jpg --out output
"""
import argparse
import os
import numpy as np

import utils
import filters
import inverse
from utils import banner


def stage1_load(path):
    banner("STAGE 1: Real-world data -> matrix representation",
           "An image is an H x W x 3 array; each pixel is an RGB vector",
           "Turn raw pixels into numbers a matrix can act on")
    img = utils.load_image(path)
    print("Loaded:", path)
    print("Image:", utils.image_info(img))
    pixels = img.reshape(-1, 3).T
    print(f"Pixel matrix P has shape {pixels.shape}  (3 rows = R,G,B ; one column per pixel)")
    print("First 3 pixel columns (RGB, 0-1):\n", np.round(pixels[:, :3], 3))
    return img


def stage2_selftest(img):
    banner("STAGE 2: Pipeline self-test (linear transformation A @ P)",
           "A 3x3 matrix acts on every pixel's RGB vector",
           "Prove the vectorised apply_filter() is correct before anyone relies on it")
    # Test 1: identity filter must return the image unchanged
    same = utils.apply_filter(img, np.eye(3))
    assert np.allclose(same, img), "Identity filter changed the image!"
    print("Identity matrix leaves the image unchanged ............ PASS")

    # Test 2: vectorised result == slow pixel-by-pixel loop
    A = filters.FILTERS["sepia"]
    crop = img[:20, :20]
    fast = utils.apply_filter(crop, A, clip=False)
    slow = np.zeros_like(crop)
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            slow[i, j] = A @ crop[i, j]
    assert np.allclose(fast, slow)
    print("Vectorised filter == per-pixel loop (20x20 crop) ...... PASS")

    # Test 3: save/load round trip (only 8-bit rounding error allowed)
    utils.save_image(img, "output/_roundtrip.png")
    back = utils.load_image("output/_roundtrip.png")
    mse, mx = utils.error_between(img, back)
    assert mx <= 0.5 / 255 + 1e-9
    print(f"Save -> load round trip (max error {mx:.4f}) ........... PASS")
    os.remove("output/_roundtrip.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="images/photo.jpg")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        fallback = "images/sample.png"
        print(f"[!] {args.image} not found -> using synthetic {fallback}. "
              f"Put a REAL photo at {args.image} for the final run.")
        args.image = utils.make_sample_image(fallback)

    os.makedirs(args.out, exist_ok=True)
    img = stage1_load(args.image)
    utils.save_image(img, f"{args.out}/original.png")
    stage2_selftest(img)

    banner("STAGE 3: Filters and matrix multiplication (Person 2)",
           "Filter = 3x3 matrix; composition = matrix product",
           "Create colour effects; show B@A applies A then B")
    filtered = filters.run(img, args.out)

    banner("STAGE 4: Inverse, determinant, rank, conditioning (Person 3)",
           "A^-1 undoes A only if det(A) != 0 (rank 3)",
           "Recover originals; show which filters lose information")
    inverse.run(img, args.out, filtered)

    print(f"\nDone. All outputs are in '{args.out}/'.")


if __name__ == "__main__":
    main()
