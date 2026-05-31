import numpy as np

def mask2rle(img: np.ndarray) -> str:
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    return ' '.join(str(x) for x in runs)

def rle2mask(rle_str: str, shape) -> np.ndarray:
    if not rle_str or rle_str == 'nan' or (isinstance(rle_str, float) and np.isnan(rle_str)):
        return np.zeros(shape, dtype=np.uint8)
    
    s = rle_str.split()
    starts = np.array(s[0::2], dtype=int) - 1  # Convert 1-based to 0-based
    lengths = np.array(s[1::2], dtype=int)
    
    flat_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat_mask[start:start + length] = 1
        
    return flat_mask.reshape(shape[1], shape[0]).T  # Reshape and transpose back

def main():
    # 1. Simple manual verification
    print("--- 1. Simple Manual Test ---")
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 1] = 1
    mask[2, 2] = 1
    mask[3, 2] = 1
    
    print("Original Mask:\n", mask)
    rle = mask2rle(mask)
    print("Encoded RLE (1-based):", rle)
    
    # Let's decode it
    decoded = rle2mask(rle, mask.shape)
    print("Decoded Mask:\n", decoded)
    assert np.array_equal(mask, decoded), "Error: Roundtrip failed on manual mask!"
    print("Manual roundtrip test passed!")
    
    # 2. Large random test
    print("\n--- 2. Random Large-Scale Test ---")
    np.random.seed(42)
    for i in range(100):
        h, w = np.random.randint(100, 1000, size=2)
        # Create random binary mask with some structure
        mask_rand = (np.random.rand(h, w) > 0.85).astype(np.uint8)
        
        rle_rand = mask2rle(mask_rand)
        decoded_rand = rle2mask(rle_rand, (h, w))
        
        assert np.array_equal(mask_rand, decoded_rand), f"Error: Random roundtrip failed at iteration {i}!"
        
    print("All 100 random roundtrip tests passed successfully!")

if __name__ == '__main__':
    main()
