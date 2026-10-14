"""
inverse.py -- PERSON 3
Matrix inverses, determinant/rank/condition analysis, image recovery,
a hand-coded 3x3 cofactor inverse, and inverse-of-product verification.

Keeps the required interface: run(img, out_dir, filtered=None)

Changes made during Person 1's review are marked with "# FIX".
"""

import os
import numpy as np
from utils import apply_filter, save_image, error_between, show_side_by_side
from filters import FILTERS


def inverse_3x3_by_hand(A, tol=1e-12):
    """Compute a 3x3 inverse using the determinant/cofactor/adjugate formula."""
    A = np.asarray(A, dtype=np.float64)
    if A.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got {A.shape}")

    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]

    # Cofactor matrix.
    C = np.array([
        [e * i - f * h, -(d * i - f * g), d * h - e * g],
        [-(b * i - c * h), a * i - c * g, -(a * h - b * g)],
        [b * f - c * e, -(a * f - c * d), a * e - b * d],
    ], dtype=np.float64)

    det = float(a * C[0, 0] + b * C[0, 1] + c * C[0, 2])
    if abs(det) <= tol:
        raise np.linalg.LinAlgError("Matrix is singular; inverse does not exist.")

    # A^-1 = transpose(cofactor matrix) / determinant.
    return C.T / det


def to_8bit(img):
    """Simulate saving an image as a normal 8-bit file and loading it back:
    clip to [0, 1], then round to the nearest of 256 levels."""
    return np.rint(np.clip(img, 0.0, 1.0) * 255.0) / 255.0


def run(img, out_dir, filtered=None):
    """Analyze filter matrices and test recovery/composition."""
    os.makedirs(out_dir, exist_ok=True)
    print("\nPERSON 3: MATRIX INVERSION AND IMAGE RECOVERY")

    print("\n1) Determinant, rank, condition number, and inverse")
    inverses = {}
    conds = {}
    for name, A in FILTERS.items():
        det = float(np.linalg.det(A))
        rank = int(np.linalg.matrix_rank(A))
        cond = float(np.linalg.cond(A))
        print(f"\n{name}:")
        print("  Matrix:\n", A)
        print(f"  determinant = {det:.6g}")
        print(f"  rank = {rank}")
        # FIX: a singular matrix has an infinite condition number; NumPy's
        # huge number (e.g. 6e49) is just rounding noise and confuses people.
        if rank < 3:
            print("  condition number = infinite (singular matrix)")
        else:
            print(f"  condition number = {cond:.6g}")

        if rank == 3 and np.isfinite(cond):
            inv_np = np.linalg.inv(A)
            inverses[name] = inv_np
            conds[name] = cond
            print("  inverse (NumPy):\n", np.array2string(inv_np, precision=6))
        else:
            print("  No inverse: this matrix is singular (information is lost).")

    hand_name = "mix"
    hand_inv = inverse_3x3_by_hand(FILTERS[hand_name])
    np_inv = np.linalg.inv(FILTERS[hand_name])
    print(f"\n2) Hand-calculated cofactor inverse for '{hand_name}':")
    print(hand_inv)
    print("Hand inverse matches NumPy:",
          bool(np.allclose(hand_inv, np_inv, rtol=1e-8, atol=1e-8)))
    assert np.allclose(hand_inv, np_inv, rtol=1e-8, atol=1e-8)

    print("\n3) Image recovery, exact case (no clipping, no rounding)")
    for name, A in FILTERS.items():
        if name not in inverses:
            print(f"  {name}: skipped - singular matrix cannot recover original RGB.")
            continue
        transformed_raw = apply_filter(img, A, clip=False)
        recovered = apply_filter(transformed_raw, inverses[name], clip=False)
        mse, max_err = error_between(img, recovered)
        print(f"  {name}: MSE={mse:.3g}, max abs error={max_err:.3g}")
    print("  -> Every invertible filter is undone perfectly (errors are ~1e-12 or less).")

    print("\n4) Image recovery, realistic case (filtered image clipped to [0,1] "
          "and saved as 8-bit)")
    print(f"  {'filter':<11}{'cond':>11}{'MSE':>14}{'max abs err':>14}")
    recovery_errors = {}
    recovered_imgs = {}
    for name, A in FILTERS.items():
        if name not in inverses:
            continue
        if filtered is not None and name in filtered:
            saved = filtered[name]                 # already clipped by filters.run
        else:
            saved = apply_filter(img, A, clip=True)
        recovered = apply_filter(to_8bit(saved), inverses[name], clip=False)
        mse, max_err = error_between(img, recovered)
        recovery_errors[name] = (mse, max_err)
        recovered_imgs[name] = recovered
        print(f"  {name:<11}{conds[name]:>11.4g}{mse:>14.4g}{max_err:>14.4g}")
        save_image(recovered, os.path.join(out_dir, f"recovered_{name}.png"))

    print("  Why the errors differ:")
    print("   - swap_rb / identity only move numbers around, so nothing is lost.")
    print("   - saturate / brightness push some values outside [0, 1], which")
    print("     clipping cuts off; that information cannot come back.")
    print("   - sepia has a huge condition number, so even tiny rounding and")
    print("     clipping errors are blown up by A^-1: the recovery is garbage.")

    # Side-by-side figure for the report: original + every recovered image.
    names = [n for n in recovered_imgs if n != "identity"]
    show_side_by_side(
        [img] + [recovered_imgs[n] for n in names],
        ["original"] + [f"recovered ({n})" for n in names],
        save_to=os.path.join(out_dir, "recovery_comparison.png"),
        show=False,
    )

    gray = FILTERS.get("grayscale")
    if gray is not None:
        print("\n5) Singular grayscale case")
        print(f"  determinant = {np.linalg.det(gray):.6g}")
        print(f"  rank = {np.linalg.matrix_rank(gray)}")
        # FIX: a concrete example makes "information is lost" visible.
        c1 = np.array([0.9, 0.3, 0.3])   # a reddish colour
        c2 = np.array([0.3, 0.9, 0.3])   # a greenish colour
        print(f"  Two different colours: {c1} and {c2}")
        print(f"  grayscale({c1}) = {gray @ c1}")
        print(f"  grayscale({c2}) = {gray @ c2}")
        print("  Same output for different inputs, so no inverse can tell them apart.")

    print("\n6) Inverse of a product: (B @ A)^-1 = A^-1 @ B^-1")
    A = FILTERS["swap_rb"]
    B = FILTERS["saturate"]
    BA = B @ A
    left = np.linalg.inv(BA)
    right = np.linalg.inv(A) @ np.linalg.inv(B)
    print("(B @ A)^-1:\n", left)
    print("A^-1 @ B^-1:\n", right)
    print("Product inverse identity PASS:", bool(np.allclose(left, right)))
    assert np.allclose(left, right)

    print("\nPERSON 3 COMPLETE: inversion, recovery, singularity, and composition checks passed.")
    return {
        "inverses": inverses,
        "recovery_errors": recovery_errors,
        "hand_inverse": hand_inv,
    }

