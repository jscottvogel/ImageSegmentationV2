import torch
from competitive_model import FloodNetCompetitiveModel

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing competitive model on {device}...")
    
    # 1. Instantiate model
    model = FloodNetCompetitiveModel(num_classes=10).to(device)
    
    # 2. Checkpoints paths
    unet_path = "model_checkpoint/FloodNet_UNet/best_unet_weights.pt"
    fcn_path = "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
    deeplab_path = "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt"
    meta_path = "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt"
    
    # 3. Load checkpoints
    model.load_checkpoints(
        unet_path=unet_path,
        fcn_path=fcn_path,
        deeplab_path=deeplab_path,
        meta_path=meta_path,
        device=device
    )
    
    # 4. Generate dummy inputs
    dummy_input = torch.randn(2, 3, 480, 640).to(device)
    
    # 5. Test training mode
    print("Testing forward pass in training mode...")
    model.train()
    logits_train = model(dummy_input)
    print(f"Training mode output shape: {logits_train.shape}")
    assert logits_train.shape == (2, 10, 480, 640), f"Expected shape (2, 10, 480, 640), got {logits_train.shape}"
    
    # 6. Test evaluation mode (with built-in TTA)
    print("Testing forward pass in evaluation mode (with built-in TTA)...")
    model.eval()
    with torch.no_grad():
        logits_eval = model(dummy_input)
    print(f"Evaluation mode output shape: {logits_eval.shape}")
    assert logits_eval.shape == (2, 10, 480, 640), f"Expected shape (2, 10, 480, 640), got {logits_eval.shape}"
    
    print("\nALL competitive model tests passed successfully!")

if __name__ == '__main__':
    test_model()
