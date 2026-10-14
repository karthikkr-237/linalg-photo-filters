"""
filters.py  --  PERSON 2 owns this file.
Starter version by Person 1 so main.py runs. Person 2: extend freely, but keep
run(img, out_dir) -> dict {name: filtered_image}, because main.py calls it.

TODO (Person 2): channel swap, brightness/tint, filter-by-filter saving,
and the composition test: apply A then B  vs  (B @ A) once  vs  (A @ B).
"""
import numpy as np
from utils import apply_filter, save_image

FILTERS = {
    "identity":  np.eye(3),
    "grayscale": np.array([[1/3, 1/3, 1/3]] * 3),
    "sepia":     np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]]),
    "swap_rb":   np.array([[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0]], dtype=float),
    "saturate":  np.diag([1.2, 0.75, 2.0]),
    "mix":       np.array([[0.70, 0.15, 0.15],
                           [0.15, 0.70, 0.15],
                           [0.15, 0.15, 0.70]]),
}


def run(img, out_dir):
    results = {}
    for name, A in FILTERS.items():
        out = apply_filter(img, A)
        save_image(out, f"{out_dir}/filtered_{name}.png")
        results[name] = out
    print(f"Applied {len(results)} filters: {', '.join(results)}")
    return results
