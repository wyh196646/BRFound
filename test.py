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
