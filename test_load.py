import torch
from unet_version import StandardUNet
from fcn_version import ResNet50FCN
from optimized_pytorch_version import CustomDeepLabV3Plus

def check_keys(model, path, device):
    state_dict = torch.load(path, map_location=device)
    if 'n_averaged' in state_dict: del state_dict['n_averaged']
    clean_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        clean_dict[k] = v
    missing, unexpected = model.load_state_dict(clean_dict, strict=False)
    print(f"[{path}] Missing: {len(missing)}, Unexpected: {len(unexpected)}")

device = torch.device('cpu')
unet = StandardUNet(num_classes=10)
fcn = ResNet50FCN(num_classes=10)
deeplab = CustomDeepLabV3Plus(num_classes=10)

check_keys(unet, "model_checkpoint/FloodNet_UNet/final_swa_smoothed_unet.pt", device)
check_keys(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
check_keys(deeplab, "model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt", device)
