import torch
from synergistic_model import FloodNetSynergisticNet

def test_synergistic_model():
    print("Testing FloodNetSynergisticNet forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    model.eval()
    
    dummy_input = torch.randn(2, 3, 480, 640).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_input)
        
    print("Output keys returned:", list(outputs.keys()))
    
    # Assertions
    expected_shape = (2, 10, 480, 640)
    for k, v in outputs.items():
        print(f"Key '{k}' output shape: {v.shape}")
        assert v.shape == expected_shape, f"Expected {expected_shape} for key {k}, got {v.shape}"
        
    print("\nALL synergistic model tests passed successfully!")

if __name__ == '__main__':
    test_synergistic_model()
