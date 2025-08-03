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
