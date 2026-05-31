import torch

def inspect(path):
    print(f"\nInspecting: {path}")
    state = torch.load(path, map_location='cpu')
    keys = list(state.keys())
    print("Total keys:", len(keys))
    # print some sample keys
    print("Sample keys:")
    for k in keys[:15]:
        print("  ", k)
    
    # count keys by prefix
    prefixes = {}
    for k in keys:
        parts = k.split('.')
        prefix = parts[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print("Prefixes count:")
    for pref, cnt in prefixes.items():
        print(f"  {pref}: {cnt}")

inspect("model_checkpoint/FloodNet_Synergistic/best_synergistic_weights_old.pt")
inspect("model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt")
