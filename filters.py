"""
filters.py -- PERSON 2
Filter matrices, filtered image saving, and composition test.
Keeps run(img, out_dir) -> dict for main.py and inverse.py.
"""

import numpy as np
from utils import (
    apply_filter,
    save_image,
    error_between,
    show_side_by_side,
)

# Luminance weights (ITU-R BT.601): how bright each colour looks to the human eye.
LUMA = np.array([0.299, 0.587, 0.114])


def saturation_matrix(s=1.5):
    """3x3 matrix that changes colour saturation.

    Saturation moves each pixel away from (or toward) its own gray value:
        new = gray + s * (pixel - gray)
    and gray = L @ pixel (L = the grayscale matrix), so
        S = s*I + (1 - s)*L        <- a plain 3x3 matrix!

    s = 0 -> grayscale (singular), s = 1 -> identity, s > 1 -> more vivid.
    Its inverse is the same formula with 1/s (for s != 0).
    FIX: this used to be a separate function outside FILTERS, which broke
    inverse.py (KeyError 'saturate'). Saturation IS a matrix, so it belongs here.
    """
    L = np.tile(LUMA, (3, 1))
    return s * np.eye(3) + (1.0 - s) * L


# All filters act on column RGB vectors: new_pixel = A @ old_pixel
FILTERS = {
    "identity": np.eye(3),

    # Every row is the same, so R, G and B all become the same gray value.
    "grayscale": np.tile(LUMA, (3, 1)),

    "sepia": np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ]),

    "swap_rb": np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=float),

    # Saturation boost (s = 1.5), written as a matrix.
    "saturate": saturation_matrix(1.5),

    "mix": np.array([
        [0.70, 0.15, 0.15],
        [0.15, 0.70, 0.15],
        [0.15, 0.15, 0.70],
    ]),

    # Brightness scaling: increase all RGB channels by 10%
    "brightness": 1.10 * np.eye(3),
}


def run(img, out_dir):
    """Apply and save each filter, then test matrix composition."""

    results = {}

    # Part 1: Apply each individual filter
    print("\nApplying individual filters:")

    for name, A in FILTERS.items():
        out = apply_filter(img, A, clip=True)
        save_image(out, f"{out_dir}/filtered_{name}.png")
        results[name] = out
        print(f"  Saved filtered_{name}.png")

    print(f"Applied {len(results)} filters: {', '.join(results)}")

    # Part 2: Composition test
    # A = red/blue swap; B = sepia
    A = FILTERS["swap_rb"]
    B = FILTERS["sepia"]

    print("\nMATRIX COMPOSITION TEST")
    print("A = swap red and blue")
    print("B = sepia tone")

    # Apply A first, then B. No intermediate clipping.
    after_A = apply_filter(img, A, clip=False)
    sequential = apply_filter(after_A, B, clip=False)

    # Applying B @ A once must produce the same result.
    BA = B @ A
    combined = apply_filter(img, BA, clip=False)

    # Reverse order: A @ B
    AB = A @ B
    reversed_order = apply_filter(img, AB, clip=False)

    # Compare numerical results before clipping or saving.
    mse_match, max_match = error_between(sequential, combined)
    mse_reverse, max_reverse = error_between(
        sequential, reversed_order
    )

    print("\nProduct B @ A (A first, then B):")
    print(BA)

    print("\nProduct A @ B (B first, then A):")
    print(AB)

    print("\nSequential vs B @ A:")
    print(f"  Mean squared error = {mse_match:.12g}")
    print(f"  Maximum absolute error = {max_match:.12g}")

    print("\nSequential vs A @ B:")
    print(f"  Mean squared error = {mse_reverse:.12g}")
    print(f"  Maximum absolute error = {max_reverse:.12g}")

    assert np.allclose(sequential, combined), (
        "Composition test failed: sequential != B @ A"
    )

    assert not np.allclose(sequential, reversed_order), (
        "Choose matrices that do not commute."
    )

    print("\nPASS: Applying A then B equals applying B @ A.")
    print("PASS: Reversing the matrix order changes the result.")

    # Save all three composition images.
    save_image(
        sequential,
        f"{out_dir}/composition_A_then_B.png",
    )
    save_image(
        combined,
        f"{out_dir}/composition_BA_once.png",
    )
    save_image(
        reversed_order,
        f"{out_dir}/composition_AB_once.png",
    )

    # Side-by-side visual comparison.
    show_side_by_side(
        [sequential, combined, reversed_order],
        ["A then B", "B @ A once", "A @ B once"],
        save_to=f"{out_dir}/composition_comparison.png",
        show=False,  # FIX: plt.show() blocks main.py until the window is closed
    )

    print(f"\nComposition images saved in {out_dir}/")
    return results
