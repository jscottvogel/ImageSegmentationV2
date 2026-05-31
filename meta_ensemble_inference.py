import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"  # Silence noisy MIOpen warnings to prevent log flooding
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. Base Architectures
from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig as Config_DL
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

# 2. Meta-Learner Architecture
from train_meta_ensemble import MetaEnsemble, load_weights

id2color = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
}

def decode_segmap(image, nc=10):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = id2color[l][0]
        g[idx] = id2color[l][1]
        b[idx] = id2color[l][2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def mask2rle(img: np.ndarray) -> str:
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    return ' '.join(str(x) for x in runs)

def run_meta_inference():
    print("Initialize Meta-Learner Stacked Inference Pipeline...")
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    SUBMISSION_PATH = "meta_ensemble_submission.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load exactly as we did in the training script
    print("Loading Base Models...")
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    unet = StandardUNet(num_classes=10).to(device)
    fcn = ResNet50FCN(num_classes=10).to(device)
    
    dl_path = "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt"
    if not os.path.exists(dl_path): dl_path = "model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt"
    deeplab = load_weights(deeplab, dl_path, device)
    
    un_path = "model_checkpoint/FloodNet_UNet/best_unet_weights.pt"
    if not os.path.exists(un_path): un_path = "model_checkpoint/FloodNet_UNet/final_swa_smoothed_unet.pt"
    unet = load_weights(unet, un_path, device)
    
    fn_path = "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
    fcn = load_weights(fcn, fn_path, device)

    # Initialize and load Meta-Learner
    print("Loading Meta-Learner 1x1 Convolutions...")
    meta_model = MetaEnsemble(unet, fcn, deeplab).to(device)
    meta_weights = "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt"
    
    if not os.path.exists(meta_weights):
        print(f"CRITICAL ERROR: {meta_weights} not found. Please run train_meta_ensemble.py first!")
        return
        
    meta_model.meta_layer.load_state_dict(torch.load(meta_weights, map_location=device))
    meta_model.eval()
    print("Successfully mounted the Meta-Learner Stack.")

    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    submission_data = []
    os.makedirs("visualizations_meta", exist_ok=True)

    print(f"Running Meta-Ensemble Inference on {len(test_images)} targets...")
    for idx, img_path in enumerate(tqdm(test_images)):
        filename = os.path.basename(img_path).replace('.jpg', '')
        base_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = base_img.shape[:2]
        
        img_tensor = cv2.resize(base_img, (Config_DL.IMG_WIDTH, Config_DL.IMG_HEIGHT))
        img_tensor = img_tensor.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_tensor = (img_tensor - mean) / std
        
        img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)

        with torch.no_grad():
            # The Meta-Learner handles dynamically blending all 3 networks based on its learned 1x1 spatial confidence
            fused_logits = meta_model(img_tensor)
            # Softmax to get probabilities (since F.interpolate on logits is equal, but doing it on probabilities makes the CPU resizing consistent)
            fused_probs = F.softmax(fused_logits, dim=1)
            probs_cpu = fused_probs.squeeze(0).cpu().numpy()
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(probs_cpu[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            
        for class_id in range(Config_DL.NUM_CLASSES):
            binary_mask = (pred_labels == class_id).astype(np.uint8)
            encoded_string = mask2rle(binary_mask)
            submission_data.append([f"{filename}_{class_id:02d}", encoded_string])
            
        if idx % 10 == 0:
            mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
            has_gt = os.path.exists(mask_path)
            
            fig = plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(base_img)
            plt.title(f"Original: {filename}")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            if has_gt:
                plt.imshow(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))
                plt.title("Ground Truth Mask")
            else:
                plt.text(0.5, 0.5, "No GT matches test-set", ha='center', va='center')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(decode_segmap(pred_labels))
            plt.title("Meta-Learner Stacked Prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"visualizations_meta/blend_check_{filename}.jpg", bbox_inches='tight')
            fig.clf()
            plt.close(fig)
            
        # Free GPU memory and empty cache
        torch.cuda.empty_cache()
        import gc; gc.collect()
            
    submission_df = pd.DataFrame(submission_data, columns=["IMG_ID", "EncodedString"])
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSUCCESS! Meta-Ensemble Hybrid CSV Generated AT: {SUBMISSION_PATH}")

if __name__ == '__main__':
    run_meta_inference()
