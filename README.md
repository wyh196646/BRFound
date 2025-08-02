# BRFound

## A Clinically Feasible Whole Slide Foundation Model for Breast Oncology

[[`Model`](https://huggingface.co/Microgle/BRFound)] [[`Paper`](https://aka.ms/BRFound)] [[`BibTeX`](#Citation)]

Yuhao Wang*, Fei Ren*, Baizhi Wang*, Yunjie Gu, Qingsong Yao, Han Li, Fenghe Tang, Qingpeng Kong, Rongsheng Wang, Xin Luo, Zikang Xu, Yijun Zhou, Wei Ba, Xueyuan Zhang, Kun Zhang, Zhigang Song, Zihang Jiang, Xiuwu Bian, Rui Yan*, S. Kevin Zhou* (*Equal Contribution)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 📢 News

<!-- ### September 2024
- **Embeddings**: We are pleased to share a new notebook, showcasing embedding visualization for BRFound. Check out the [notebook](https://github.com/BRFound/BRFound/blob/main/demo/BRFound_pca_visualization_timm.ipynb) to get started.

![BRFound Embedding Visualization](https://github.com/BRFound/BRFound/blob/main/images/BRFound_embedding_visualization.png)

### July 2024
- **WSI Preprocessing/Tiling**: We are pleased to share new preprocessing guide for BRFound. This guide provides a walkthrough on setting up the environment and preprocessing WSI files for BRFound. Check out the [guide](https://github.com/BRFound/BRFound/blob/main/BRFound/preprocessing/preprocessing.md) to get started.

### June 2024
- **New Demo Notebook Available**: We have prepared a new notebook for the walkthrough of the BRFound model. This notebook provides a detailed demonstration of how to load and run the pretrained model. You can check it out [here](https://github.com/BRFound/BRFound/blob/main/demo/run_BRFound.ipynb). -->

### May 2024
- **Initial Model and Code Release**: We are excited toe release BRFound BRFound model and its code is now available. 
## Model Overview

<p align="center">
    <img src="images/BRFound_overview.png" width="90%"> <br>

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

## Inference

The BRFound model consists of a tile encoder, that extracts local patterns at patch level, and a slide encoder, that outputs representations at slide level. This model can be used in both tile-level and slide-level tasks. When doing inference at the slide level, we recommend following this pipeline: (1) Tile the whole slide into N image tiles, with the coordinates of each tile. (2) Get the embeddings for each tile using our tile encoder. (3) Pass the N image tile embeddings and their coordinates into the slide encoder, to get slide level representations.

<!-- ### Inference with the tile encoder

First, load BRFound tile encoder:

```Python
import timm
from PIL import Image
from torchvision import transforms
import torch

tile_encoder = timm.create_model("hf_hub:BRFound/BRFound", pretrained=True)

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
```

Running inference to extract tile level features:

```Python
img_path = "images/prov_normal_000_1.png"
sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

tile_encoder.eval()
with torch.no_grad():
    output = tile_encoder(sample_input).squeeze()
```

**

### Inference with the slide encoder

To inference with our slide encoder, we need both the tile embeddings and their coordinates as input. First, let's load the BRFound slide encoder:

```Python
import BRFound

slide_encoder = BRFound.slide_encoder.create_model("hf_hub:BRFound/BRFound", "BRFound_slide_enc12l768d", 1536)
```

Run the inference to get the slide level embeddings:

```Python
slide_encoder.eval()
with torch.no_grad():
    output = slide_encoder(tile_embed, coordinates).squeeze()
```


**Note** Older versions of timm have compatibility issues. Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.


## Fine-tuning

### Tile-Level Linear Probing Example Using PCam Dataset

For your convenience, we provide the pre-extracted embeddings for the PCam dataset. You can download them from this [link](https://huggingface.co/datasets/BRFound/BRFound-tile-embeddings/tree/main). Note that the file size is 2GB.

There is no need to unzip this file.

To run the fine-tuning experiment, execute the following script:
```sh
bash scripts/run_pcam.sh data/BRFound_PCam_embeddings.zip
```

### Slide-Level Fine-Tuning Example Using PANDA Dataset

For your convenience, we provide the pre-extracted embeddings for the PANDA dataset. You can download them from this [link](https://huggingface.co/datasets/BRFound/BRFound-tile-embeddings/tree/main). Note that the file size is 32GB. Please unzip this file.

```sh
unzip -n data/BRFound_PANDA_embeddings.zip -d data/
```

To run the fine-tuning experiment, execute the following script:
```sh
bash scripts/run_panda.sh data/BRFound_PANDA_embeddings/h5_files
```

## Sample Data Download

A sample de-identified subset of the Prov-Path data can be accessed from these links [[1](https://zenodo.org/records/10909616), [2](https://zenodo.org/records/10909922)].

## Model Uses

### Intended Use
The data, code, and model checkpoints are intended to be used solely for (I) future research on pathology foundation models and (II) reproducibility of the experimental results reported in the reference paper. The data, code, and model checkpoints are not intended to be used in clinical care or for any clinical decision-making purposes.

### Primary Intended Use
The primary intended use is to support AI researchers reproducing and building on top of this work. BRFound should be helpful for exploring pre-training, and encoding of digital pathology slides data.

### Out-of-Scope Use
**Any** deployed use case of the model --- commercial or otherwise --- is out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are intended *for research use only* and not intended for deployed use cases.

## Usage and License Notices

The model is not intended or made available for clinical use as a medical device, clinical support, diagnostic tool, or other technology intended to be used in the diagnosis, cure, mitigation, treatment, or prevention of disease or other conditions. The model is not designed or intended to be a substitute for professional medical advice, diagnosis, treatment, or judgment and should not be used as such.  All users are responsible for reviewing the output of the developed model to determine whether the model meets the user’s needs and for validating and evaluating the model before any clinical use.

## Acknowledgements

We would like to express our gratitude to the authors and developers of the exceptional repositories that this project is built upon: DINOv2, MAE, Timm, and TorchScale. Their contributions have been invaluable to our work. -->

## Citation
If you find BRFound useful for your your research and applications, please cite using this BibTeX:

<!-- ```bibtex
@article{xu2024BRFound,
  title={A whole-slide foundation model for digital pathology from real-world data},
  author={Xu, Hanwen and Usuyama, Naoto and Bagga, Jaspreet and Zhang, Sheng and Rao, Rajesh and Naumann, Tristan and Wong, Cliff and Gero, Zelalem and González, Javier and Gu, Yu and Xu, Yanbo and Wei, Mu and Wang, Wenhui and Ma, Shuming and Wei, Furu and Yang, Jianwei and Li, Chunyuan and Gao, Jianfeng and Rosemon, Jaylen and Bower, Tucker and Lee, Soohee and Weerasinghe, Roshanthi and Wright, Bill J. and Robicsek, Ari and Piening, Brian and Bifulco, Carlo and Wang, Sheng and Poon, Hoifung},
  journal={Nature},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
``` -->

