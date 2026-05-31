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

@torch.no_grad()
def multiscale_inference(model, image_tensor):
    model.eval()
    # Standard Forward Pass
    out_std = model(image_tensor)
    if isinstance(out_std, dict):
        out_std = out_std['main_output']
    
    # Horizontal Flip Pass
    img_flipped = torch.flip(image_tensor, dims=[3])
    out_flip = model(img_flipped)
    if isinstance(out_flip, dict):
        out_flip = out_flip['main_output']
    out_unflipped = torch.flip(out_flip, dims=[3])
    
    return (out_std + out_unflipped) / 2.0

from rfdetr_version import DenseTransformerSegmentation
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
    """Highly optimized RLE encoder translating dense pixel matrices to Kaggle submission formats."""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    return ' '.join(str(x) for x in runs)

def safe_morphological_cleanup(pred_labels: np.ndarray, min_area=25) -> np.ndarray:
    """
    Kaggle Grandmaster technique: Deletes isolated microscopic false-positives.
    Unlike median blur which hallucinated classes at borders, this uses Connected Components
    to strictly target isolated blobs (e.g. a 5-pixel 'vehicle' in a lake) and safely 
    reverts them to the background class without touching valid structural edges.
    """
    clean_labels = pred_labels.copy()
    for class_id in np.unique(pred_labels):
        if class_id == 0: continue  # Skip background
        
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                clean_labels[labels == i] = 0  # Revert microscopic anomaly to background
                
    return clean_labels

def ensemble_generate_submission():
    print("Initialize Advanced Multi-Model Hybrid Inference Pipeline...")
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    SUBMISSION_PATH = "hybrid_ensemble_submission.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -------------------------------------------------------------
    # ENSEMBLE SYSTEM: UNCOMMENT THE MODELS YOU WISH TO BLEND
    # -------------------------------------------------------------
    models_to_ensemble = []
    
    # 1. BASELINE CNN (DeepLabV3+ with ASAPP) - [STATUS: OPTIMIZED]
    try:
        from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig as Config_DL, TrainingConfig as Train_DL
        model_cnn = CustomDeepLabV3Plus(num_classes=Config_DL.NUM_CLASSES).to(device)
        # Disabled torch.compile() globally to prevent AMD ROCm hardware crashes
        # try:
        #     model_cnn = torch.compile(model_cnn)
        #     print("Successfully compiled DeepLabV3+ with Triton for inference.")
        # except Exception:
        #     pass
            
        # STRICTLY Load Non-SWA Weights (SWA update_bn destroys frozen ImageNet statistics)
        cnn_weight_path = os.path.join(Train_DL.CHECKPOINT_DIR, "best_deeplab_weights.pt")
        
        state_dict_cnn = torch.load(cnn_weight_path, map_location=device)
        if 'n_averaged' in state_dict_cnn: del state_dict_cnn['n_averaged']
        
        target_keys = set(model_cnn.state_dict().keys())
        is_compiled = any(k.startswith('_orig_mod.') for k in target_keys)
        
        clean_state_dict = {}
        for k, v in state_dict_cnn.items():
            k = k.replace('module.', '').replace('_orig_mod.', '')
            if is_compiled: k = '_orig_mod.' + k
            clean_state_dict[k] = v
            
        model_cnn.load_state_dict(clean_state_dict)
        model_cnn.eval()
        
        w_dl = torch.tensor([1.0, 0.9301, 0.9380, 0.9007, 0.9252, 0.9556, 0.9224, 0.8694, 0.9242, 0.9590], device=device).view(1, 10, 1, 1)
        models_to_ensemble.append(('DeepLabV3+', lambda x: F.softmax(multiscale_inference(model_cnn, x) / 3.0, dim=1), w_dl))
        print(f"Loaded Core Architecture [DeepLabV3+] gracefully.")
    except Exception as e:
        print(f"Bypassing DeepLabV3+: {e}")

    # 2. ISOLATED CNN (Standard UNet) - [STATUS: UNCOMMENT TO ALLOCATE]
    try:
        from unet_version import StandardUNet
        model_unet = StandardUNet(num_classes=10).to(device)
        # Disabled torch.compile() globally to prevent AMD ROCm hardware crashes
        # try:
        #     model_unet = torch.compile(model_unet)
        #     print("Successfully compiled StandardUNet with Triton for inference.")
        # except Exception:
        #     pass
            
        unet_weight_path = "model_checkpoint/FloodNet_UNet/best_unet_weights.pt"
        if os.path.exists(unet_weight_path):
            state_dict_unet = torch.load(unet_weight_path, map_location=device)
            if 'n_averaged' in state_dict_unet: del state_dict_unet['n_averaged']
            target_keys = set(model_unet.state_dict().keys())
            is_compiled = any(k.startswith('_orig_mod.') for k in target_keys)
            
            clean_state_unet_dict = {}
            for k, v in state_dict_unet.items():
                k = k.replace('module.', '').replace('_orig_mod.', '')
                if is_compiled: k = '_orig_mod.' + k
                clean_state_unet_dict[k] = v
                
            model_unet.load_state_dict(clean_state_unet_dict)
            model_unet.eval()
            w_unet = torch.tensor([1.0, 0.9434, 0.9526, 0.9234, 0.9359, 0.9552, 0.9333, 0.8939, 0.9356, 0.9637], device=device).view(1, 10, 1, 1)
            models_to_ensemble.append(('UNet', lambda x: F.softmax(multiscale_inference(model_unet, x) / 3.0, dim=1), w_unet))
            print(f"Loaded Core Architecture [StandardUNet] gracefully.")
        else:
            print(f"Bypassing UNet: Weights not found yet. Execute Training first.")
    except Exception as e:
        print(f"Bypassing UNet: {e}")

    # 3. ISOLATED FCN (ResNet50)
    try:
        from fcn_version import ResNet50FCN
        model_fcn = ResNet50FCN(num_classes=10).to(device)
        fcn_weight_path = "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
        if os.path.exists(fcn_weight_path):
            state_dict_fcn = torch.load(fcn_weight_path, map_location=device)
            if 'n_averaged' in state_dict_fcn: del state_dict_fcn['n_averaged']
            target_keys = set(model_fcn.state_dict().keys())
            is_compiled = any(k.startswith('_orig_mod.') for k in target_keys)
            
            clean_state_fcn_dict = {}
            for k, v in state_dict_fcn.items():
                k = k.replace('module.', '').replace('_orig_mod.', '')
                if is_compiled: k = '_orig_mod.' + k
                clean_state_fcn_dict[k] = v
                
            model_fcn.load_state_dict(clean_state_fcn_dict)
            model_fcn.eval()
            w_fcn = torch.tensor([1.0, 0.9195, 0.9385, 0.9176, 0.9298, 0.9473, 0.9254, 0.8448, 0.9257, 0.9595], device=device).view(1, 10, 1, 1)
            models_to_ensemble.append(('FCN', lambda x: F.softmax(multiscale_inference(model_fcn, x) / 3.0, dim=1), w_fcn))
            print(f"Loaded Core Architecture [ResNet50-FCN] gracefully.")
        else:
            print(f"Bypassing FCN: Weights not found yet. Execute Training first.")
    except Exception as e:
        print(f"Bypassing FCN: {e}")
        
    if len(models_to_ensemble) == 0:
        print("FATAL ERROR: No models uncommented or verified. Exiting securely...")
        return
        pass
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    if len(test_images) == 0:
        print("CRITICAL: Failed to locate test images directory!")
        return
        
    submission_data = []
    
    os.makedirs("visualizations", exist_ok=True)

    print(f"Running Multi-Architecture Hybrid Prediction on {len(test_images)} targets...")
    for idx, img_path in enumerate(tqdm(test_images)):
        filename = os.path.basename(img_path).replace('.jpg', '')
        base_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = base_img.shape[:2]
        
        # Extract native scaling
        img_tensor = cv2.resize(base_img, (Config_DL.IMG_WIDTH, Config_DL.IMG_HEIGHT))
        img_tensor = img_tensor.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_tensor = (img_tensor - mean) / std
        
        # Deploy matrix to PyTorch GPU core
        img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)

        with torch.no_grad():
            fused_ensemble_probs = torch.zeros((1, 10, Config_DL.IMG_HEIGHT, Config_DL.IMG_WIDTH), device=device)
            
            # Dynamically route matrices applying accuracy-based weights
            total_weight = sum(w for _, _, w in models_to_ensemble)
            for model_name, model_eval_func, weight in models_to_ensemble:
                fused_ensemble_probs += model_eval_func(img_tensor) * (weight / total_weight)
            
            # CPU Interpolation optimization to save 600MB VRAM per step
            probs_cpu = fused_ensemble_probs.squeeze(0).cpu().numpy()
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(probs_cpu[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            
            # KAGGLE GRANDMASTER LEVER: Obliterate isolated false-positive microscopic pixels
            pred_labels = safe_morphological_cleanup(pred_labels, min_area=50)
            
            
        for class_id in range(Config_DL.NUM_CLASSES):
            binary_mask = (pred_labels == class_id).astype(np.uint8)
            encoded_string = mask2rle(binary_mask)
            img_id_str = f"{filename}_{class_id:02d}"
            submission_data.append([img_id_str, encoded_string])
            
        # THE ULTIMATE LEVER: Generate physical pseudo-labels for Knowledge Distillation Retraining
        os.makedirs(Train_DL.PSEUDO_MSK_DIR, exist_ok=True)
        color_mask_bgr = cv2.cvtColor(decode_segmap(pred_labels), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(Train_DL.PSEUDO_MSK_DIR, f"{filename}.png"), color_mask_bgr)
            
        if idx % 10 == 0:
            mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
            has_gt = os.path.exists(mask_path)
            
            fig = plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(base_img)
            plt.title(f"Original Image: {filename}")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            if has_gt:
                gt_img = cv2.imread(mask_path)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                plt.imshow(gt_img)
                plt.title("Ground Truth Mask")
            else:
                plt.text(0.5, 0.5, "No GT matches test-set", ha='center', va='center')
                plt.title("Ground Truth Mask")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(decode_segmap(pred_labels))
            plt.title("Multi-Model Hybrid Prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"visualizations/blend_check_{filename}.jpg", bbox_inches='tight')
            fig.clf()
            plt.close(fig)
            
        # Free GPU memory and empty cache
        torch.cuda.empty_cache()
        import gc; gc.collect()
            
    print("Writing hybridized structural blocks to hard-drive CSV...")
    submission_df = pd.DataFrame(submission_data, columns=["IMG_ID", "EncodedString"])
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nSUCCESS! Multi-Model Hybrid CSV Generated AT:")
    print(f" -> {SUBMISSION_PATH}")

if __name__ == '__main__':
    ensemble_generate_submission()
