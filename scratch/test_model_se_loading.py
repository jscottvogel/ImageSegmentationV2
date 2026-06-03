import os
import torch
from optimized_pytorch_version import CustomDeepLabV3Plus
from unet_version import StandardUNet
from fcn_version import ResNet50FCN
from synergistic_model import FloodNetSynergisticNet

def clean_state_dict(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if 'n_averaged' in state:
        del state['n_averaged']
    clean = {}
    for k, v in state.items():
        clean[k.replace('module.', '').replace('_orig_mod.', '')] = v
    return clean

def test_loading():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing weight loading on device: {device}\n")
    
    # 1. DeepLabV3+
    print("Testing DeepLabV3+ with SE...")
    dl = CustomDeepLabV3Plus(num_classes=10, use_se=True).to(device)
    dl_path = "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt"
    if os.path.exists(dl_path):
        state = clean_state_dict(dl_path, device)
        dl.load_state_dict(state, strict=True)
        print("  [SUCCESS] DeepLabV3+ SE weights loaded strictly!")
    else:
        print(f"  [ERROR] File not found: {dl_path}")
        
    # 2. UNet
    print("\nTesting UNet with SE...")
    unet = StandardUNet(num_classes=10, use_se=True).to(device)
    unet_path = "model_checkpoint/FloodNet_UNet/best_unet_weights.pt"
    if os.path.exists(unet_path):
        state = clean_state_dict(unet_path, device)
        unet.load_state_dict(state, strict=True)
        print("  [SUCCESS] UNet SE weights loaded strictly!")
    else:
        print(f"  [ERROR] File not found: {unet_path}")
        
    # 3. FCN
    print("\nTesting FCN with SE...")
    fcn = ResNet50FCN(num_classes=10, use_se=True).to(device)
    fcn_path = "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
    if os.path.exists(fcn_path):
        state = clean_state_dict(fcn_path, device)
        fcn.load_state_dict(state, strict=True)
        print("  [SUCCESS] FCN SE weights loaded strictly!")
    else:
        print(f"  [ERROR] File not found: {fcn_path}")
        
    # 4. Synergistic model
    print("\nTesting Synergistic model with SE...")
    syn = FloodNetSynergisticNet(num_classes=10, use_se=True).to(device)
    syn_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(syn_path):
        state = clean_state_dict(syn_path, device)
        syn.load_state_dict(state, strict=True)
        print("  [SUCCESS] Synergistic SE weights loaded strictly!")
    else:
        print(f"  [ERROR] File not found: {syn_path}")

if __name__ == '__main__':
    test_loading()
