import cv2
import numpy as np

try:
    guide = np.random.rand(512, 512, 3).astype(np.float32)
    src = np.random.rand(512, 512).astype(np.float32)
    
    out = cv2.ximgproc.jointBilateralFilter(guide, src, d=5, sigmaColor=10, sigmaSpace=10)
    print("Joint Bilateral Filter succeeded. Shape:", out.shape)
except AttributeError:
    print("ximgproc not available. Need opencv-contrib-python")
