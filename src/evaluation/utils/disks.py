#!/usr/bin/env python3
"""
Generate floating-point grayscale images with non-overlapping, non-cropped circles.
Pixel values in [0.0, 1.0], saved as 32-bit TIFF by default.
Optional Gaussian blur and Gaussian noise applied post-render.

Usage:
    python gen_circles.py [--width W] [--height H] [--r1 R1] [--r2 R2]
                          [--n N] [--bg BG] [--fg FG] [--out PATH] [--seed S]
                          [--margin M] [--blur SIGMA] [--noise SIGMA]
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from skimage.draw import disk,ellipse
from skimage.filters import gaussian
from skimage.io import imsave
from skimage.util import random_noise


def _overlaps(
    cx: float, cy: float, r: float,
    placed: list[tuple],
    margin: float,
) -> bool:
    """Return True if (cx, cy, r) overlaps any placed circle within *margin* clearance."""
    for px, py, pr in placed:
        if math.hypot(cx - px, cy - py) < r + pr + margin:
            return True
    return False


def place_circles(
    width: int,
    height: int,
    r_min: float,
    r_max: float,
    n_circles: int,
    margin: float = 0.0,
    max_attempts: int = 10_000,
    rng: random.Random | None = None,
) -> list[tuple[float, float, float]]:
    if rng is None:
        rng = random.Random()

    placed: list[tuple[float, float, float]] = []
    attempts = 0

    while len(placed) < n_circles and attempts < max_attempts:
        r = rng.uniform(r_min, r_max)
        # Boundary margin uses r only — circles never touch the image edge.
        cx = rng.uniform(r, width - r)
        cy = rng.uniform(r, height - r)
        if not _overlaps(cx, cy, r, placed, margin):
            placed.append((cx, cy, r))
        attempts += 1

    if len(placed) < n_circles:
        print(
            f"Warning: placed {len(placed)}/{n_circles} circles "
            f"after {max_attempts} attempts. "
            f"Consider reducing n, r_max, or margin."
        )
    return placed


def render(
    circles: list[tuple[float, float, float]],
    width: int,
    height: int,
    bg: float,
    fg: float,
    blur_sigma: float,
    noise_sigma: float,
    out_path: Path,
    rng_seed: int | None,
) -> None:
    img = np.full((height, width), bg, dtype=np.float32)

    for cx, cy, r in circles:
        rr, cc = disk((cy, cx), r, shape=img.shape)
        img[rr, cc] = fg

    if blur_sigma > 0.0:
        img = gaussian(img, sigma=blur_sigma).astype(np.float32)

    if noise_sigma > 0.0:
        # random_noise expects and returns float64; cast back to float32.
        img = random_noise(
            img,
            mode="gaussian",
            var=noise_sigma ** 2,
            rng=rng_seed,
        ).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

    imsave(out_path, img)
    print(f"Saved {len(circles)} circles → {out_path}  (float32, shape {img.shape})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width",  type=int,   default=512,                  help="Image width  (px)")
    p.add_argument("--height", type=int,   default=512,                  help="Image height (px)")
    p.add_argument("--r1",     type=float, default=10.0,                 help="Min radius (px)")
    p.add_argument("--r2",     type=float, default=40.0,                 help="Max radius (px)")
    p.add_argument("--n",      type=int,   default=50,                   help="Number of circles to attempt")
    p.add_argument("--bg",     type=float, default=0.0,                  help="Background value [0.0, 1.0]")
    p.add_argument("--fg",     type=float, default=1.0,                  help="Circle fill value [0.0, 1.0]")
    p.add_argument("--margin", type=float, default=2.0,                  help="Min clearance between circle edges (px)")
    p.add_argument("--blur",   type=float, default=1.0,                  help="Gaussian blur sigma (px); 0 to disable")
    p.add_argument("--noise",  type=float, default=0.02,                 help="Gaussian noise sigma [0.0, 1.0]; 0 to disable")
    p.add_argument("--out",    type=Path,  default=Path("circles.tiff"), help="Output path (.tiff recommended)")
    p.add_argument("--seed",   type=int,   default=None,                 help="RNG seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.r1 > args.r2:
        raise ValueError(f"r1={args.r1} must be ≤ r2={args.r2}")
    if args.r2 * 2 >= min(args.width, args.height):
        raise ValueError("r2 too large relative to image dimensions")
    for name, val in (("bg", args.bg), ("fg", args.fg)):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"--{name} must be in [0.0, 1.0], got {val}")
    if args.margin < 0.0:
        raise ValueError("--margin must be ≥ 0")
    if args.blur < 0.0:
        raise ValueError("--blur sigma must be ≥ 0")
    if not 0.0 <= args.noise <= 1.0:
        raise ValueError("--noise sigma must be in [0.0, 1.0]")

    rng = random.Random(args.seed)
    circles = place_circles(
        args.width, args.height,
        args.r1, args.r2,
        args.n,
        margin=args.margin,
        rng=rng,
    )
    render(
        circles, args.width, args.height,
        args.bg, args.fg,
        blur_sigma=args.blur,
        noise_sigma=args.noise,
        out_path=args.out,
        rng_seed=args.seed,
    )


if __name__ == "__main__":
    main()