import os
import time
import torch
import torch.nn as nn
# Import models
from unet_version import StandardUNet
from optimized_pytorch_version import CustomDeepLabV3Plus
from fcn_version import ResNet50FCN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(num):
    if num >= 1e6:
        return f"{num / 1e6:.3f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    return str(num)

def benchmark_model(model_class, model_name, device, input_size=(2, 3, 480, 640), num_runs=20):
    print(f"\n--- Profiling {model_name} ---")
    
    # 1. Parameter Overhead
    model_std = model_class(use_se=False).to(device)
    model_se = model_class(use_se=True).to(device)
    
    params_std = count_parameters(model_std)
    params_se = count_parameters(model_se)
    overhead = params_se - params_std
    overhead_pct = (overhead / params_std) * 100
    
    print(f"Parameters (Standard): {format_params(params_std)} ({params_std:,})")
    print(f"Parameters (SE):       {format_params(params_se)} ({params_se:,})")
    print(f"SE Overhead:           {format_params(overhead)} (+{overhead_pct:.3f}%)")
    
    # Create dummy input
    x = torch.randn(input_size, device=device)
    
    # Warmup runs
    for _ in range(5):
        _ = model_std(x)
        _ = model_se(x)
    
    # Create dummy optimizer to clean gradients
    opt_std = torch.optim.SGD(model_std.parameters(), lr=0.01)
    opt_se = torch.optim.SGD(model_se.parameters(), lr=0.01)

    # Benchmark standard
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(num_runs):
        opt_std.zero_grad(set_to_none=True)
        out = model_std(x)
        # Check if dict or tensor
        if isinstance(out, dict):
            main_out = out['main_output']
        else:
            main_out = out
        loss = main_out.sum()
        loss.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_std = (time.time() - t0) / num_runs
    fps_std = input_size[0] / t_std
    
    # Benchmark SE
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(num_runs):
        opt_se.zero_grad(set_to_none=True)
        out = model_se(x)
        if isinstance(out, dict):
            main_out = out['main_output']
        else:
            main_out = out
        loss = main_out.sum()
        loss.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_se = (time.time() - t0) / num_runs
    fps_se = input_size[0] / t_se
    
    fps_diff_pct = ((fps_se - fps_std) / fps_std) * 100
    print(f"Throughput (Standard): {fps_std:.2f} FPS (per-step time: {t_std*1000:.2f} ms)")
    print(f"Throughput (SE):       {fps_se:.2f} FPS (per-step time: {t_se*1000:.2f} ms)")
    print(f"SE FPS Impact:         {fps_diff_pct:.2f}%")
    
    # 2. Check backward compatibility with existing checkpoint
    # Find checkpoint
    ckpt_dirs = {
        "UNet": "model_checkpoint/FloodNet_UNet/best_unet_weights.pt",
        "DeepLabV3+": "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        "FCN": "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
    }
    ckpt_path = ckpt_dirs.get(model_name)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Verifying checkpoint loading from {ckpt_path}...")
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Remove keys prefix if compiled
            clean_state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            # Load into standard model
            try:
                model_std.load_state_dict(clean_state_dict, strict=True)
                print("  [OK] Successfully loaded into standard model with strict=True.")
            except Exception as e_strict:
                print(f"  [INFO] strict=True loading failed for standard model: {str(e_strict)[:120]}...")
                missing_std, unexpected_std = model_std.load_state_dict(clean_state_dict, strict=False)
                print(f"  [OK] Loaded into standard model with strict=False (missing={len(missing_std)}, unexpected={len(unexpected_std)})")
            
            # Load into SE model
            missing_keys, unexpected_keys = model_se.load_state_dict(clean_state_dict, strict=False)
            print(f"  [OK] Successfully loaded into SE model with strict=False.")
            print(f"       Missing keys (SE-specific parameters initialized from scratch): {len(missing_keys)}")
            print(f"       Unexpected keys (should be 0): {len(unexpected_keys)}")
            if len(unexpected_keys) > 0:
                print(f"       WARNING: Unexpected keys found: {unexpected_keys}")
        except Exception as e:
            print(f"  [ERROR] Checkpoint loading failed: {e}")
    else:
        print(f"Checkpoint for {model_name} not found at {ckpt_path}, skipping loading check.")
        
    return {
        "Name": model_name,
        "Params Std": params_std,
        "Params SE": params_se,
        "Param Overhead": overhead,
        "Param Overhead Pct": overhead_pct,
        "FPS Std": fps_std,
        "FPS SE": fps_se,
        "FPS Impact Pct": fps_diff_pct
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = []
    
    # 1. Profile UNet
    results.append(benchmark_model(StandardUNet, "UNet", device))
    
    # 2. Profile DeepLabV3+
    results.append(benchmark_model(CustomDeepLabV3Plus, "DeepLabV3+", device))
    
    # 3. Profile FCN
    results.append(benchmark_model(ResNet50FCN, "FCN", device))
    
    # Print Summary Table
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    headers = ["Model", "Params (Std)", "Params (SE)", "Param Overhead", "FPS (Std)", "FPS (SE)", "FPS Impact"]
    rows = []
    for r in results:
        rows.append([
            r["Name"],
            format_params(r["Params Std"]),
            format_params(r["Params SE"]),
            f"+{r['Param Overhead Pct']:.3f}%",
            f"{r['FPS Std']:.1f}",
            f"{r['FPS SE']:.1f}",
            f"{r['FPS Impact Pct']:.1f}%"
        ])
    
    # Custom format
    col_widths = [15, 15, 15, 15, 12, 12, 12]
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        print(" | ".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths)))
    print("="*80)

if __name__ == '__main__':
    main()
