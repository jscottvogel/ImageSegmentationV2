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

def multiscale_meta_inference(meta_model, img_tensor, device):
    unet = meta_model.unet
    fcn = meta_model.fcn
    deeplab = meta_model.deeplab
    meta_layer = meta_model.meta_layer
    
    _, _, h, w = img_tensor.shape
    
    # Standard scale only (1.0) with horizontal flip for extremely fast, stable TTA
    scales = [1.0]
    scale_weights = [1.0]
    
    fused_probs_u = torch.zeros((1, 10, h, w), device=device)
    fused_probs_f = torch.zeros((1, 10, h, w), device=device)
    fused_probs_d = torch.zeros((1, 10, h, w), device=device)
    
    for scale, weight in zip(scales, scale_weights):
        scaled_size = (int(h * scale), int(w * scale))
        scaled_img = F.interpolate(img_tensor, size=scaled_size, mode='bilinear', align_corners=False)
        
        # Standard Forward Pass
        out_u_std = F.softmax(unet(scaled_img)['main_output'], dim=1)
        out_f_std = F.softmax(fcn(scaled_img)['main_output'], dim=1)
        out_d_std = F.softmax(deeplab(scaled_img)['main_output'], dim=1)
        
        fused_probs_u += F.interpolate(out_u_std, size=(h, w), mode='bilinear', align_corners=False) * weight
        fused_probs_f += F.interpolate(out_f_std, size=(h, w), mode='bilinear', align_corners=False) * weight
        fused_probs_d += F.interpolate(out_d_std, size=(h, w), mode='bilinear', align_corners=False) * weight
        
        # TTA Horizontal Flip Pass
        scaled_img_flipped = torch.flip(scaled_img, dims=[3])
        out_u_flip = F.softmax(unet(scaled_img_flipped)['main_output'], dim=1)
        out_f_flip = F.softmax(fcn(scaled_img_flipped)['main_output'], dim=1)
        out_d_flip = F.softmax(deeplab(scaled_img_flipped)['main_output'], dim=1)
        
        out_u_unflip = torch.flip(out_u_flip, dims=[3])
        out_f_unflip = torch.flip(out_f_flip, dims=[3])
        out_d_unflip = torch.flip(out_d_flip, dims=[3])
        
        fused_probs_u += F.interpolate(out_u_unflip, size=(h, w), mode='bilinear', align_corners=False) * weight
        fused_probs_f += F.interpolate(out_f_unflip, size=(h, w), mode='bilinear', align_corners=False) * weight
        fused_probs_d += F.interpolate(out_d_unflip, size=(h, w), mode='bilinear', align_corners=False) * weight
        
    # Standard + Flipped H
    fused_probs_u = fused_probs_u / 2.0
    fused_probs_f = fused_probs_f / 2.0
    fused_probs_d = fused_probs_d / 2.0
    
    stacked_logits = torch.cat([fused_probs_u, fused_probs_f, fused_probs_d], dim=1)
    return meta_layer(stacked_logits)

def run_meta_inference_tta():
    print("Initialize Meta-Learner Stacked Inference Pipeline with MST/TTA...")
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    SUBMISSION_PATH = "meta_ensemble_submission_tta.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    os.makedirs("visualizations_meta_tta", exist_ok=True)

    print(f"Running Meta-Ensemble Inference with TTA/MST on {len(test_images)} targets...")
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
            fused_logits = multiscale_meta_inference(meta_model, img_tensor, device)
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
            plt.title("Meta-Learner Stacked TTA Prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"visualizations_meta_tta/blend_check_{filename}.jpg", bbox_inches='tight')
            fig.clf()
            plt.close(fig)
            
        # Free GPU memory and empty cache
        torch.cuda.empty_cache()
        import gc; gc.collect()
            
    submission_df = pd.DataFrame(submission_data, columns=["IMG_ID", "EncodedString"])
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSUCCESS! Meta-Ensemble TTA/MST CSV Generated AT: {SUBMISSION_PATH}")

if __name__ == '__main__':
    run_meta_inference_tta()
