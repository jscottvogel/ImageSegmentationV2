import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, FloodNetPyTorchDataset, id2color
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

class MetaDataset(FloodNetPyTorchDataset):
    def __getitem__(self, idx: int):
        img, label = self._load(idx)
        augmented = self.transform(image=img, mask=label)
        img, label = augmented['image'], augmented['mask'].numpy()
        dummy_edge = torch.zeros((1, DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), dtype=torch.float32)
        return img, {'main_output': torch.tensor(label, dtype=torch.long), 'edge_output': dummy_edge}

class MetaEnsemble(nn.Module):
    def __init__(self, unet, fcn, deeplab, num_classes=10):
        super().__init__()
        self.unet = unet
        self.fcn = fcn
        self.deeplab = deeplab
        
        # Freeze base models completely
        for model in [self.unet, self.fcn, self.deeplab]:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()

        # Meta Learner (3 models * 10 classes = 30 channels)
        self.meta_layer = nn.Conv2d(in_channels=30, out_channels=num_classes, kernel_size=1)
        
        # Mathematically initialize to a perfect Average Ensemble baseline
        nn.init.constant_(self.meta_layer.weight, 0)
        nn.init.constant_(self.meta_layer.bias, 0)
        for i in range(num_classes):
            self.meta_layer.weight.data[i, i] = 1.0 / 3.0
            self.meta_layer.weight.data[i, i + num_classes] = 1.0 / 3.0
            self.meta_layer.weight.data[i, i + 2*num_classes] = 1.0 / 3.0

    def train(self, mode=True):
        super().train(mode)
        # Force base models to stay in evaluation mode (frozen batch statistics)
        self.unet.eval()
        self.fcn.eval()
        self.deeplab.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            out_u = F.softmax(self.unet(x)['main_output'], dim=1)
            out_f = F.softmax(self.fcn(x)['main_output'], dim=1)
            
            # DeepLab multiscale or single scale? Use single scale for speed during meta-training
            out_d = F.softmax(self.deeplab(x)['main_output'], dim=1)
            
        stacked_logits = torch.cat([out_u, out_f, out_d], dim=1)
        return self.meta_layer(stacked_logits)

def load_weights(model, path, device):
    state_dict = torch.load(path, map_location=device)
    if 'n_averaged' in state_dict:
        del state_dict['n_averaged']
    clean_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        clean_dict[k] = v
    model.load_state_dict(clean_dict, strict=False)
    return model

def train_meta():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Meta-Learner Training on {device}...")
    
    # 1. Load Base Models
    print("Loading DeepLabV3+...")
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab_path = "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt"
    if not os.path.exists(deeplab_path): 
        deeplab_path = "model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt"
    deeplab = load_weights(deeplab, deeplab_path, device)
    
    print("Loading UNet...")
    unet = StandardUNet(num_classes=10).to(device)
    unet_path = "model_checkpoint/FloodNet_UNet/best_unet_weights.pt"
    if not os.path.exists(unet_path): 
        unet_path = "model_checkpoint/FloodNet_UNet/final_swa_smoothed_unet.pt"
    unet = load_weights(unet, unet_path, device)
    
    print("Loading FCN...")
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn_path = "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
    fcn = load_weights(fcn, fcn_path, device)
    
    # 2. Build Meta-Ensemble
    meta_model = MetaEnsemble(unet, fcn, deeplab).to(device)
    
    # 3. Mount Dataloader
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    ps_img = sorted(glob.glob(os.path.join("/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images", "*.jpg")))
    ps_msk = sorted(glob.glob(os.path.join("confident_pseudo_masks", "*.png")))
    
    if len(ps_msk) > 0:
        print(f"Injecting {len(ps_msk)} high-confidence pseudo-masks into meta-training!")
        tr_img.extend(ps_img)
        tr_msk.extend(ps_msk)
        
    # Zip and shuffle with fixed seed to select 300 representative samples
    combined = list(zip(tr_img, tr_msk))
    import random
    random.seed(42)
    random.shuffle(combined)
    combined_sub = combined[:300]
    tr_img_sub, tr_msk_sub = zip(*combined_sub)
    tr_img_sub, tr_msk_sub = list(tr_img_sub), list(tr_msk_sub)
    
    print(f"Subsampled to {len(tr_img_sub)} images for fast meta-training.")
    dataset = MetaDataset(tr_img_sub, tr_msk_sub, DatasetConfig.NUM_CLASSES, id2color, apply_aug=False, use_mosaic=False)
    # Use batch size 4 for speed and memory stability (fits 12GB VRAM easily)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    # 4. Train only the meta layer
    optimizer = torch.optim.AdamW(meta_model.meta_layer.parameters(), lr=0.001)
    from optimized_pytorch_version import class_weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    EPOCHS = 10 # Increased to 10 epochs for full convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    os.makedirs("model_checkpoint/FloodNet_Meta", exist_ok=True)
    
    for epoch in range(EPOCHS):
        meta_model.train() # Sets the meta layer to train (base models remain frozen)
        epoch_loss = 0.0
        
        for b_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            labels = targets['main_output'].to(device)
            
            optimizer.zero_grad()
            preds = meta_model(images)
            
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if b_idx % 25 == 0:
                print(f"Meta-Epoch {epoch+1}/{EPOCHS} | Batch {b_idx}/{len(loader)} | Loss: {loss.item():.4f}")
                
        print(f"--> Epoch {epoch+1} Avg Loss: {epoch_loss/len(loader):.4f}")
        scheduler.step()
        
    torch.save(meta_model.meta_layer.state_dict(), "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt")
    print("Meta-Learner weights successfully saved!")

if __name__ == '__main__':
    train_meta()
