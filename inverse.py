"""
inverse.py  --  PERSON 3 owns this file.
Starter version by Person 1 so main.py runs. Person 3: replace the body of run().
Keep run(img, out_dir, filtered=None). `filtered` is the dict from filters.run().

TODO (Person 3): np.linalg.inv per filter, one 3x3 inverse by hand, det + rank,
apply A^-1 to filtered images (use apply_filter(..., clip=False) to test exact
recovery), error vs original (utils.error_between), grayscale singular case,
np.linalg.cond, and (BA)^-1 = A^-1 B^-1.
"""
import numpy as np
from filters import FILTERS


def run(img, out_dir, filtered=None):
    print("Determinant / rank / condition number of each filter:")
    for name, A in FILTERS.items():
        print(f"  {name:10s} det = {np.linalg.det(A): .3e}   "
              f"rank = {np.linalg.matrix_rank(A)}   cond = {np.linalg.cond(A):.3e}")
    print("(starter only -- Person 3 adds the inversion + recovery steps)")
