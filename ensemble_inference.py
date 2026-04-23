import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

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
        from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig as Config_DL, TrainingConfig as Train_DL, multiscale_inference
        model_cnn = CustomDeepLabV3Plus(num_classes=Config_DL.NUM_CLASSES).to(device)
        # Disabled torch.compile() globally to prevent AMD ROCm hardware crashes
        # try:
        #     model_cnn = torch.compile(model_cnn)
        #     print("Successfully compiled DeepLabV3+ with Triton for inference.")
        # except Exception:
        #     pass
            
        cnn_weight_path = os.path.join(Train_DL.CHECKPOINT_DIR, "final_swa_smoothed_weights.pt")
        if not os.path.exists(cnn_weight_path): 
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
        
        # Add securely decoupled prediction wrapper
        models_to_ensemble.append(('DeepLabV3+', lambda x: F.softmax(multiscale_inference(model_cnn, x), dim=1)))
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
            target_keys = set(model_unet.state_dict().keys())
            is_compiled = any(k.startswith('_orig_mod.') for k in target_keys)
            
            clean_state_unet_dict = {}
            for k, v in state_dict_unet.items():
                k = k.replace('module.', '').replace('_orig_mod.', '')
                if is_compiled: k = '_orig_mod.' + k
                clean_state_unet_dict[k] = v
                
            model_unet.load_state_dict(clean_state_unet_dict)
            model_unet.eval()
            models_to_ensemble.append(('UNet', lambda x: F.softmax(model_unet(x)['main_output'], dim=1)))
            print(f"Loaded Core Architecture [StandardUNet] gracefully.")
        else:
            print(f"Bypassing UNet: Weights not found yet. Execute Training first.")
    except Exception as e:
        print(f"Bypassing UNet: {e}")
        
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
            
            # Dynamically route matrices cleanly sequentially
            for model_name, model_eval_func in models_to_ensemble:
                fused_ensemble_probs += model_eval_func(img_tensor)
                
            # Physically average probabilities 
            fused_ensemble_probs /= len(models_to_ensemble)
            
            # Snap dimensions exactly back to the physical pixels native to drone
            logits_reshaped = F.interpolate(fused_ensemble_probs, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_labels = torch.argmax(logits_reshaped, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
        for class_id in range(Config_DL.NUM_CLASSES):
            binary_mask = (pred_labels == class_id).astype(np.uint8)
            encoded_string = mask2rle(binary_mask)
            img_id_str = f"{filename}_{class_id:02d}"
            submission_data.append([img_id_str, encoded_string])
            
        if idx % 10 == 0:
            mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
            has_gt = os.path.exists(mask_path)
            
            plt.figure(figsize=(18, 6))
            
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
            plt.close()
            
    print("Writing hybridized structural blocks to hard-drive CSV...")
    submission_df = pd.DataFrame(submission_data, columns=["IMG_ID", "EncodedString"])
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nSUCCESS! Multi-Model Hybrid CSV Generated AT:")
    print(f" -> {SUBMISSION_PATH}")

if __name__ == '__main__':
    ensemble_generate_submission()
