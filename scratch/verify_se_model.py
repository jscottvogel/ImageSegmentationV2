import torch
from synergistic_model import FloodNetSynergisticNet

def verify():
    print("Instantiating FloodNetSynergisticNet with Channel Attention...")
    model = FloodNetSynergisticNet(num_classes=10)
    print("Model instantiated successfully!")
    
    # Create a dummy input tensor matching the (batch_size, channels, height, width) of the dataset
    x = torch.randn(2, 3, 480, 640)
    print(f"Input shape: {x.shape}")
    
    print("Running forward pass...")
    outputs = model(x)
    
    print("Output keys:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
        
    assert outputs['main_output'].shape == (2, 10, 480, 640), "Output shape mismatch!"
    print("Forward pass shape checks PASSED!")
    
    print("Testing backward pass on main output...")
    loss = outputs['main_output'].sum()
    loss.backward()
    print("Backward pass PASSED! Gradients successfully propagated.")

if __name__ == '__main__':
    verify()
