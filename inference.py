"""Extract one BRFound slide embedding from precomputed patch features."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.cluster import KMeans

from src.slide_transformer import vit_base


def sample_tokens(features, coords, ratio=0.25, clusters=8, seed=42, max_tokens=4000):
    """Reproducible cluster-wise local sampling; preserve feature-coordinate pairs."""
    if not 0 < ratio <= 1 or clusters < 1 or max_tokens < 1:
        raise ValueError('ratio must be in (0, 1]; clusters and max_tokens must be positive')
    rng = np.random.default_rng(seed)
    labels = KMeans(n_clusters=min(clusters, len(features)), random_state=seed,
                    n_init=10).fit_predict(features)
    chosen = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        anchor = rng.choice(indices)
        distances = np.sum((features[indices] - features[anchor]) ** 2, axis=1)
        count = max(1, int(len(indices) * ratio))
        chosen.extend(indices[np.argsort(distances, kind='stable')[:count]])
    chosen = np.asarray(chosen)
    if len(chosen) > max_tokens:
        chosen = np.sort(rng.choice(chosen, max_tokens, replace=False))
    return features[chosen], coords[chosen]


def encode_slide(features_path, weights_path, output_path, device='cpu', seed=42,
                 ratio=0.25, clusters=8, max_tokens=4000):
    with h5py.File(features_path, 'r') as handle:
        features = np.asarray(handle['features'], dtype=np.float32)
        coords = np.asarray(handle['coords'], dtype=np.float32)
    if features.ndim != 2 or len(features) == 0 or coords.shape != (len(features), 2):
        raise ValueError('Expected nonempty features [N, D] and matching coords [N, 2]')
    if not np.isfinite(features).all() or not np.isfinite(coords).all() or (coords < 0).any():
        raise ValueError('Inputs must be finite; coordinates must be nonnegative')
    # The inference release is a plain tensor state dictionary, without training
    # objects or an optimizer. Strict loading catches incompatible feature encoders.
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    if 'patch_embed.proj.weight' not in state:
        raise ValueError('Use slide_encoder_inference.pth from Microgle/BRFound')
    expected_dim = state['patch_embed.proj.weight'].shape[1]
    if features.shape[1] != expected_dim:
        raise ValueError(f'Checkpoint expects {expected_dim}-D patch features, got {features.shape[1]}')
    model = vit_base(slide_embedding_size=expected_dim, dynamic_pos_embed=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    features, coords = sample_tokens(features, coords, ratio, clusters, seed, max_tokens)
    features = torch.from_numpy(features).unsqueeze(0).to(device)
    coords = torch.from_numpy(coords).unsqueeze(0).to(device)
    mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=device)
    with torch.inference_mode():
        embedding = model(features, coords, mask).float().cpu().numpy()
    if embedding.shape != (1, 768) or not np.isfinite(embedding).all():
        raise RuntimeError('Unexpected slide embedding shape or non-finite output')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as handle:
        np.save(handle, embedding)
    print(f'Saved {tuple(embedding.shape)} embedding from {features.shape[1]} tokens to {output_path}')
    return embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, help='HDF5 with features and coords datasets')
    parser.add_argument('--weights', required=True, help='slide_encoder_inference.pth')
    parser.add_argument('--output', required=True, help='Output NumPy .npy path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratio', type=float, default=0.25)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=4000)
    args = parser.parse_args()
    encode_slide(args.features, args.weights, args.output, args.device, args.seed,
                 args.ratio, args.clusters, args.max_tokens)


if __name__ == '__main__':
    main()
