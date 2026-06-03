import numpy as np
import time

def mask2rle_orig(img: np.ndarray) -> str:
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(runs.astype(str))

def mask2rle_opt(img: np.ndarray) -> str:
    pixels = img.T.ravel()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(map(str, runs))

mask = np.zeros((3000, 4000), dtype=np.uint8)
mask[500:1500, 1000:3000] = 1
np.random.seed(42)
noise = np.random.rand(3000, 4000) < 0.01
mask = np.logical_or(mask, noise).astype(np.uint8)

print("Running 10 loops of RLE encoding on 3000x4000 mask...")
t0 = time.time()
for _ in range(10):
    r1 = mask2rle_orig(mask)
t1 = time.time()
print(f"Original mask2rle: {t1 - t0:.4f} seconds")

t0 = time.time()
for _ in range(10):
    r2 = mask2rle_opt(mask)
t1 = time.time()
print(f"Optimized mask2rle: {t1 - t0:.4f} seconds")

assert r1 == r2
