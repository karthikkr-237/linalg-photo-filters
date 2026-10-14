# Photo Filters with Matrix Multiplication and Inversion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karthikkr-237/linalg-photo-filters/blob/main/linalg_photo_filters_colab.ipynb)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-report%20Completed-green)

A Linear Algebra mini project (PES University, Department of CSE). A colour photo is a matrix of RGB pixels. We apply photo filters as **3x3 matrix multiplication**, combine them with **matrix products**, and try to undo them with **matrix inverses**. The project shows which filters can be reversed, which cannot, and why some reversible ones still fail in practice.

## The idea in 30 seconds

Every pixel is a column vector `p = [R, G, B]^T`. A filter is a 3x3 matrix `A`:

```
new_pixel = A @ old_pixel
```

Stack all `H x W` pixels as the columns of one `3 x (H*W)` matrix `P`. A single product `A @ P` filters the whole image at once.

- **Composition:** "apply `A`, then `B`" is the single matrix `B @ A`. Order matters: `A @ B` gives a different image.
- **Inversion:** if `det(A) != 0` (rank 3), then `A^-1 @ (A @ p) = p` recovers the original pixel.
- **Information loss:** grayscale has rank 1. Different colours map to the same gray, so it cannot be undone.
- **Conditioning:** sepia is invertible on paper, but its condition number is about 1.7e4, so tiny rounding and clipping errors are blown up and recovery fails in practice.

## Workflow stages covered

Following the course guidelines (real-world data to application output):

| Stage | What we do |
|---|---|
| Real-world data | A real photograph (see [Dataset](#dataset)) |
| Matrix representation | Image as an `H x W x 3` array, reshaped to a `3 x (H*W)` pixel matrix |
| Linear transformation | Seven filters as 3x3 matrices, `A @ P` |
| Matrix multiplication | Composition test: `A` then `B` equals `B @ A`, and `A @ B` differs |
| Structure of the space | Determinant, rank, invertibility of each filter |
| Inversion and recovery | `A^-1` applied to the filtered images, error versus the original |
| Application output | Filtered images, recovered images, conditioning analysis |

## Repository structure

```
.
├── utils.py                          # image <-> matrix pipeline (Person 1)
├── main.py                           # runs every stage, saves to output/ (Person 1)
├── filters.py                        # filter matrices and composition test (Person 2)
├── inverse.py                        # inverses, det/rank/cond, recovery (Person 3)
├── linalg_photo_filters_colab.ipynb  # one-click Colab notebook for the demo
├── images/                           # input photo: photo.jpg
├── output/                           # generated results
└── README.md
```

## Quick start

**Google Colab (easiest):** click the Open in Colab badge, then Runtime, Run all. Run the upload cell if you want to use your own photo.

**Local:**

```bash
git clone https://github.com/karthikkr-237/linalg-photo-filters.git
cd linalg-photo-filters
pip install numpy pillow matplotlib
python main.py                         # uses images/photo.jpg
python main.py --image my_photo.jpg --out output
```

If `images/photo.jpg` is missing, `main.py` falls back to a synthetic test image and prints a warning. Use a real photo for the final run.

## What `main.py` runs

| Stage | Module | What it does |
|---|---|---|
| 1 | `main.py` | Loads the image and shows its pixel matrix `P` (3 x H*W) |
| 2 | `main.py` | Self-tests: identity leaves the image unchanged, vectorised filter equals a per-pixel loop, save/load round-trips |
| 3 | `filters.py` | Applies the seven filters, then runs the composition test |
| 4 | `inverse.py` | Determinant, rank, condition number, inverses, hand-coded 3x3 inverse, recovery tests, singular case, `(BA)^-1 = A^-1 B^-1` |

## The filters

| Filter | Determinant | Rank | Condition number | Invertible? |
|---|---|---|---|---|
| identity | 1 | 3 | 1 | Yes |
| grayscale | 0 | 1 | infinite | **No** (information is lost) |
| sepia | 1.21e-7 | 3 | about 1.67e4 | Yes, but very fragile |
| swap_rb | -1 | 3 | 1 | Yes |
| saturate | 1.8 | 3 | about 2.67 | Yes |
| mix | 0.3025 | 3 | about 1.82 | Yes |
| brightness | 1.331 | 3 | 1 | Yes |

## Recovery results (what to expect)

`inverse.py` tests recovery two ways:

1. **Exact case** (no clipping, no rounding): every invertible filter is undone perfectly, with errors around 1e-12 or smaller. This shows the maths is right.
2. **Realistic case** (filtered image clipped to 0-1 and saved as 8-bit): this is what actually happens to saved images.

| Filter | Realistic recovery |
|---|---|
| identity, swap_rb | Perfect: values are only moved around |
| mix | Near perfect: only 8-bit rounding error |
| saturate, brightness | Loses detail where clipping cut values above 1 |
| sepia | Fails badly: the huge condition number amplifies tiny errors into noise |
| grayscale | Impossible: no inverse exists |

Exact error values depend on the photo, so see the printed output and `output/recovery_comparison.png`.

## Output files (`output/`)

| File | Contents |
|---|---|
| `original.png` | The input image |
| `filtered_<name>.png` | Each filter applied (7 files) |
| `composition_A_then_B.png`, `composition_BA_once.png`, `composition_AB_once.png` | The composition test images; the first two match, the third differs |
| `composition_comparison.png` | The three composition images side by side |
| `recovered_<name>.png` | Realistic recovery (clipped, 8-bit) for each invertible filter |
| `recovery_comparison.png` | Original next to every recovered image |

## Conventions

Everyone working on this repo follows these, so the modules fit together:

1. An image is a `float64` array of shape `(H, W, 3)` with values in `[0, 1]`. Do not convert to 8-bit between steps.
2. A filter acts on a column vector: `new_pixel = A @ old_pixel`.
3. "Apply `A`, then `B`" equals the single matrix `B @ A`.
4. Use `apply_filter(img, A, clip=False)` to test exact inverses; use the default `clip=True` only for display and saving.
5. `main.py` expects `filters.run(img, out_dir)` to return `{name: filtered_image}` and `inverse.run(img, out_dir, filtered)`.
6. Plots that are only saved use `show=False`, so `main.py` never stops to open a window.

### Core functions (`utils.py`)

| Function | Purpose |
|---|---|
| `load_image(path)` | Read a file into a float `(H, W, 3)` array in `[0, 1]` |
| `apply_filter(img, A, clip=True)` | Reshape to `3 x (H*W)`, compute `A @ P`, reshape back |
| `save_image(img, path)` | Convert to 8-bit and save, creating folders as needed |
| `error_between(a, b)` | Return (MSE, max absolute error) between two images |
| `show_side_by_side(images, titles, save_to, show)` | Display and/or save images in one row |

## Project status

| Part | Owner | Status |
|---|---|---|
| Image pipeline, `main.py`, Colab notebook | Person 1 | Done |
| Filters and composition test (`filters.py`) | Person 2 | Done |
| Inverses, determinant, rank, recovery, conditioning (`inverse.py`) | Person 3 | Done |
| Report, hand calculations, conclusions | Person 4 | In progress |

Remaining before the presentation: add the real photo, fill in the placeholders below, run once in Colab, and finish the report. For the hand calculation, use the `mix` matrix, since that is the one `inverse.py` verifies by hand.

## Team

| Name | Role |
|---|---|
|KARTHIK K R| Pipeline and command line |
|J DEVI PRASHANTH| Filters and multiplication |
|HARSHA N| Inversion |
|KAUSHIK D ACHARYA| Report, math and testing |

## Dataset

Input image: Image.jpg by Wikipedia.

## Results

_Add before/after images from `output/` here once the full run is complete._

## Course context

Mini project for the Linear Algebra course, PES University. Evaluation covers a demo (execution, concepts, output of each stage) and a viva based on Concept, Purpose, Outcome.
