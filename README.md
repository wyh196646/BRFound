# BRFound

## A Clinically Feasible Whole Slide Foundation Model for Breast Oncology

[[`Model`](https://huggingface.co/Microgle/BRFound)] [[`Paper`]] 

Yuhao Wang, Fei Ren, Baizhi Wang*, Yunjie Gu, Qingsong Yao, Han Li, Fenghe Tang, Qingpeng Kong, Rongsheng Wang, Xin Luo, Zikang Xu, Yijun Zhou, Wei Ba, Xueyuan Zhang, Kun Zhang, Zhigang Song, Zihang Jiang, Xiuwu Bian, Rui Yan, S. Kevin Zhou* (*Cooresponding Author)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


### August 2025
- **Initial Model and Code Release**: We are excited toe release BRFound BRFound model and its code is now available. 
## Model Overview

<p align="center">
    <img src="images/Model_Overview.jpg" width="90%"> <br>

  *Overview of BRFound model architecture*

</p>

## Install


1. Download our repository and open the BRFound
```
git clone https://github.com/wyh196646/BRFound
cd BRFound
```

2. Install BRFound and its dependencies

```Shell
conda env create -f environment.yaml
conda activate BRFound
pip install -e .
```

## Model Download

The BRFound models can be accessed from [HuggingFace Hub](https://huggingface.co/Microgle/BRFound).


## Inference with BRFound
During the model development process, we extensively referenced the open-source slide-level foundational model Gigapath. As a result, our data preprocessing steps are largely consistent with those of Gigapath. For more details, please refer to the [Gigapath](https://github.com/prov-gigapath/prov-gigapath.git) repository.


### Whole Slide Image Preprocessing

### Runing Inference with the Patch Encoder of BRFound
```
import torch
from easydict import EasyDict
from torchvision import transforms
from PIL import Image
import sys
import os
from src import build_model_from_cfg
from src.vision_transformer import vit_base
from src.utils import load_pretrained_weights



def get_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(image_path, model, device='cuda'):
    transform = get_transform()

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image_tensor)

    return features.cpu().numpy()

def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model

if __name__ == "__main__":
    
    config = EasyDict({
        'student': EasyDict({
            'arch': 'vit_base', 
            'patch_size': 16, 
            'drop_path_rate': 0.3,  
            'layerscale': 1.0e-05,  
            'drop_path_uniform': True,  
            'pretrained_weights': '',  
            'ffn_layer': 'mlp',  
            'block_chunks': 4,  
            'qkv_bias': True,  
            'proj_bias': True,  
            'ffn_bias': True,  
            'num_register_tokens': 0,  
            'interpolate_antialias': False,  
            'interpolate_offset': 0.1  
        }),
        'crops': EasyDict({
            'global_crops_size': 224,
        })    
    })
        
    weights_path = './weights/patch_encoder.pth'
    image_path = './images/patch_1.png'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if files exist
    if not os.path.exists(weights_path):
        print(f"Warning: weights file not found at {weights_path}")
    if not os.path.exists(image_path):
        print(f"Warning: image file not found at {image_path}")
    
    # Only proceed if imports were successful and files exist

    model = build_model_for_eval(config, weights_path)
    features = extract_features(image_path, model, device=device)
    print(f"Extracted features shape: {features.shape}")
```

### Runing Inference with the Slide  Encoder of BRFound

```
import torch
import numpy as np
from sklearn.cluster import KMeans
import h5py
from torch.utils.data import Dataset, DataLoader
from src.slide_transformer import vit_base

# Slide dataset class
class SlideDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.features, self.coords = self._read_h5(h5_path)

    @staticmethod
    def _read_h5(h5_path):
        with h5py.File(h5_path, 'r') as f:
            features = f['features'][:]
            coords = f['coords'][:]
        return features, coords

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.coords[idx]
    
class ClusterSelector:
    def __init__(self, num_clusters=8, selection_ratio=0.25):
        self.num_clusters = num_clusters
        self.selection_ratio = selection_ratio

    def select_patches(self, features, coords):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        selected_features = []
        selected_coords = []

        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_features = features[cluster_indices]
            cluster_coords = coords[cluster_indices]

            # Select a random seed patch
            seed_idx = np.random.choice(len(cluster_features))
            seed_feature = cluster_features[seed_idx]

            # Calculate distances and select closest 25%
            distances = np.linalg.norm(cluster_features - seed_feature, axis=1)
            num_select = max(1, int(len(cluster_features) * self.selection_ratio))
            nearest_indices = distances.argsort()[:num_select]

            selected_features.append(cluster_features[nearest_indices])
            selected_coords.append(cluster_coords[nearest_indices])
            
        selected_features = np.concatenate(selected_features, axis=0)
        selected_coords = np.concatenate(selected_coords, axis=0)
        return selected_features,selected_coords

# Load pretrained model
def load_model(weights_path, device='cuda'):
    model = vit_base(slide_embedding_size=768,return_all_tokens=False)
    state_dict = torch.load(weights_path, map_location=device,weights_only=False)
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

# Extract slide-level features
def extract_slide_features(h5_path, weights_path, device='cuda'):
    dataset = SlideDataset(h5_path)
    features, coords = dataset.features, dataset.coords

    selector = ClusterSelector(num_clusters=8, selection_ratio=0.25)
    # Ensure features and coords are aligned
    assert features.shape[0] == coords.shape[0], f"Features and coords must have same first dimension, got {features.shape[0]} and {coords.shape[0]}"
    selected_features, selected_coords = selector.select_patches(features, coords)

    model = load_model(weights_path, device)
    model.to(device)
    with torch.no_grad():
        selected_features = torch.tensor(selected_features).unsqueeze(0).to(device) 
        selected_coords = torch.tensor(selected_coords).unsqueeze(0).to(device)
        masks = torch.zeros(selected_features.shape[0], 
        selected_features.shape[1], 
        dtype=torch.bool).to(device)
        slide_features = model(torch.tensor(selected_features), torch.tensor(selected_coords), masks).to(device).cpu().numpy()

    return slide_features

# Usage example
if __name__ == "__main__":
    h5_path = './images/sample.h5'
    weights_path = './weights/slide_encoder.pth'

    features = extract_slide_features(h5_path, weights_path)
    print("Extracted slide features shape:", features.shape)

```


## Acknowledgements

We would like to express our gratitude to the authors and developers of the exceptional repositories that this project is built upon: GigaPath, Donov2 and UNI Their contributions have been invaluable to our work.

