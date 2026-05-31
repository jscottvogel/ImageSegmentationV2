import os
import torch
from optimized_pytorch_version import CustomDeepLabV3Plus
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_keys(model_class, path, name):
    print(f"\n--- Checking {name} from {path} ---")
    if not os.path.exists(path):
        print("Path does not exist!")
        return
    state = torch.load(path, map_location=device, weights_only=True)
    print("Keys in checkpoint:", len(state.keys()))
    
    clean_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    print("Cleaned keys in checkpoint:", len(clean_state.keys()))
    
    model = model_class(num_classes=10)
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print("Missing keys (strict=False):", len(missing))
    print("Unexpected keys (strict=False):", len(unexpected))
    if len(missing) > 0:
        print("First 10 missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First 10 unexpected keys:", unexpected[:10])

check_keys(CustomDeepLabV3Plus, 'model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt', 'DeepLab Best')
check_keys(CustomDeepLabV3Plus, 'model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt', 'DeepLab SWA')
check_keys(StandardUNet, 'model_checkpoint/FloodNet_UNet/best_unet_weights.pt', 'UNet Best')
check_keys(StandardUNet, 'model_checkpoint/FloodNet_UNet/final_swa_smoothed_unet.pt', 'UNet SWA')
check_keys(ResNet50FCN, 'model_checkpoint/FloodNet_FCN/best_fcn_weights.pt', 'FCN Best')
check_keys(ResNet50FCN, 'model_checkpoint/FloodNet_FCN/final_swa_smoothed_fcn.pt', 'FCN SWA')
