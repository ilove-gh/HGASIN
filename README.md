# HGASIN
A PyTorch implementation of HGASIN "Heterogeneity-Based Graph Attribute Segmentation and Integration Network for Improved Mild Cognitive Impairment Detection".

## Abstract
<p align="justify">
Mild cognitive impairment (MCI), a condition featuring cognitive decline, is commonly considered a precursor of Alzheimer's disease. For MCI detection, using graph convolutional neural networks to model and analyze brain networks has become a popular technique. However, most existing methods work with a homogeneous brain network, which is easily affected by the inferior quality of heterogeneous connectivity presentation between regions of interest (ROIs) and hard to capture complex relationships and real information flow. To overcome this challenge, this paper proposes a novel Heterogeneity-based Graph Attribute Segmentation and Integration Network (HGASIN) for MCI detection. Firstly, diffusion tensor imaging and functional magnetic resonance imaging are leveraged to co-construct the functional-to-structural brain network and divide it into fine-grained areas according to the connection pattern: left cerebral hemisphere, inter-hemispheric, and right cerebral hemisphere, which are regarded as homogeneity, heterogeneity, and homogeneity subnetworks, respectively. Secondly, HGASIN discriminatively learns all subnetworks' features with different attributes using a specific encoder and comprehensively integrates these features with an adaptive attention mechanism to achieve the whole brain-level representation. Particularly, HGASIN incorporates a graph diffusion mechanism to reconstruct the structural richness of brain networks and reveal potential network relationships. Finally, the experimental study indicates that HGASIN can significantly improve the performance of MCI detection. Codes will be made available upon paper acceptance.
</p>

## Dependencies
- python 3.8.18
- pytorch 2.1.0
- cuda 11.8
- torch-geometric 1.6.2

## Code Architecture
```
|── process                 # load datasets scripts
|── utils                   # Common useful modules
|── models                  # models
|── GDCModel                # Graph Diffusion Convolution model
|  └── layers               # code for layers
|  └── models               # code for models
|── config.yaml             # Configuration file for the GDC model
└── train.py                # basic trainner and hyper-parameter
```
##  Train 
```
python train.py
```

## Citation
```
...
```