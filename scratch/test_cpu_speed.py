import os
import time
import torch
import torch.nn.functional as F
from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel

def test_cpu():
    device = torch.device('cpu')
    print("Initializing models on CPU...")
    start_time = time.time()
    
    syn_model = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(syn_weights):
        state = torch.load(syn_weights, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        syn_model.load_state_dict(state)
    syn_model.eval()
    
    meta_model = FloodNetCompetitiveModel(num_classes=10).to(device)
    meta_model.load_checkpoints(
        unet_path="model_checkpoint/FloodNet_UNet/best_unet_weights.pt",
        fcn_path="model_checkpoint/FloodNet_FCN/best_fcn_weights.pt",
        deeplab_path="model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        meta_path="model_checkpoint/FloodNet_Meta/meta_layer_weights.pt",
        device=device
    )
    meta_model.eval()
    
    print(f"Models loaded in {time.time() - start_time:.2f} seconds.")
    
    # Create a dummy batch of size 4
    x = torch.randn(4, 3, 480, 640).to(device)
    
    print("Running 1 step of inference on CPU...")
    start_time = time.time()
    with torch.no_grad():
        out_syn = syn_model(x)
        p_syn_main = F.softmax(out_syn['main_output'], dim=1).numpy()
        p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).numpy()
        p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).numpy()
        
        meta_logits = meta_model(x)
        p_meta = F.softmax(meta_logits, dim=1).numpy()
        
    print(f"Inference completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    test_cpu()
