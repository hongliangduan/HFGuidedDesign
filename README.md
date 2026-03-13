# HFGuidedDesign
De Novo Design of Cyclic Peptide Binders via Structure-Guided Discrete Diffusion

## Model Framework

<p align="center">
  <img src="./HFGuidedDesign.png">
</p>

## Installation

### Prerequisites

Before installing **HFGuidedDesign**, please make sure to install an external structure prediction model.

We recommend installing **HighFold**:

HighFold GitHub:
https://github.com/hongliangduan/HighFold

HFGuidedDesign is designed as a flexible framework. In addition to HighFold, other structure prediction models (e.g., Boltz-2, AlphaFold3, etc.) can also be integrated as external structure evaluators to guide the diffusion process.

### Installation HFGuidedDesign

```bash
conda create -n HFGuidedDesign python=3.9 -y
conda activate HFGuidedDesign
pip install -r requirements.txt
```

## Pretrained Weights

You can download the  weights from:
```bash
[https://zenodo.org/records/18768564]
```
After downloading, place the checkpoint files into the following directory:/checkpoints


## Training Discrete Diffusion model

```bash
python /models/discrete_diffusion_peptides.py
python /models/discrete_diffusion_complexes.py
```

## Using HFGuidedDesign model

```bash
python /models/hfguideddesign.py
```

## Copyright and License

This project is governed by the terms of the MIT License. Prior to utilization, kindly review the LICENSE document for comprehensive details and compliance instructions.

## Version History

- v1.0.0
