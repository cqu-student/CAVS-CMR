# Taming Modality in Continual Audio-Visual Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2510.17234-b31b1b)](https://arxiv.org/abs/2510.17234)
![TMLR 2026](https://img.shields.io/badge/TMLR-2026-purple)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green)

This repository provides the official PyTorch implementation for **CMR**, a rehearsal-based continual learning framework for **Audio-Visual Segmentation (AVSS)**.  
CMR tackles multi-modal semantic drift and co-occurrence confusion in continual AVSS.

---

## 🪵 TODO List

- [x] Release core AVSS codebase
- [x] Complete README documentation
- [x] Document data and checkpoint paths
- [ ] Add more detailed usage notes if needed

---

## 🔥 What's New

- **2026-07-02**: README reorganized into a project-homepage style.
- **2026-07-02**: Repository now emphasizes AVSS codebase, data preparation, and run commands.
- **2025-10-17**: 📄 Paper released on arXiv.

---

## 🧠 CMR: Collision-based Memory Rehearsal for Continual Audio-Visual Segmentation

Official PyTorch implementation of **CMR**, accepted at **TMLR 2026**.

### 📌 Abstract

Audio-visual segmentation (AVS) aims to localize and segment sounding objects in video by jointly modeling audio and visual signals.  
Extending AVS to a continual learning setting introduces two core challenges:

1. **Multi-modal semantic drift** across sequential tasks.
2. **Co-occurrence confusion**, where the model overfits spurious audio-visual correlations.

We propose **CMR (Collision-based Memory Rehearsal)**, a rehearsal-based framework that addresses these challenges through:

- 🎯 **Multi-modal Sample Selection**: identifies representative and diverse samples across audio-visual modalities for memory construction.
- 🔁 **Collision-based Rehearsal**: resolves co-occurrence confusion by strategically replaying samples that expose inter-class collisions.
- 🧩 **Continual AVSS Pipeline**: a full sequential learning pipeline built on AVSBench for reproducible experimentation.

---

## 🏗️ Architecture

### 🎯 Multi-modal Sample Selection
Selects informative and representative exemplars by jointly considering audio and visual feature distributions to build a compact memory buffer.

### 🔁 Collision-based Memory Rehearsal
Replays carefully selected samples that expose co-occurrence conflicts, preventing the model from exploiting spurious audio-visual correlations across tasks.

### 🧩 Continual AVSS Training Pipeline
Integrates the above components into a sequential task training loop, enabling the model to learn new categories without catastrophically forgetting prior ones.

---

## 📊 Results

Results are reported on the **AVSBench AVSS benchmark** under the continual learning setting.  
Please refer to the paper for full quantitative comparisons.

---

## 🚀 Get Started

### 1) Installation

```bash
git clone https://github.com/cqu-student/CAVS-CMR.git
cd CAVS-CMR
pip install -r requirements.txt
```

### 2) Data Preparation

1. Download the AVSBench dataset and place it under `avsbench_data/`.
2. Download pretrained backbones and place them under `pretrained_backbones/`.
3. Update data/backbone paths in `avs_scripts/avss/config.py`.

Example layout:

```text
CAVS-CMR/
├── avsbench_data/
│   ├── Multi-sources/
│   └── ...
├── pretrained_backbones/
│   └── ...
├── avs_scripts/
│   └── avss/
│       ├── config.py
│       ├── train.py
│       ├── test.py
│       ├── color_dataloader.py
│       ├── model/
│       └── utils/
└── preprocess_scripts/
```

---

## 🏋️ Train

```bash
cd avs_scripts/avss
bash train.sh
```

Key paths to configure inside `train.sh` or `config.py`:

| Parameter       | Description                                  |
|----------------|----------------------------------------------|
| `data_root`     | Path to `avsbench_data/`                     |
| `backbone_path` | Path to pretrained backbone weights          |
| `output_dir`    | Directory to save checkpoints and logs       |
| `task_split`    | Continual task split configuration           |

---

## 🧪 Test (Evaluation)

```bash
cd avs_scripts/avss
bash test.sh
```

Key paths to configure inside `test.sh` or `config.py`:

| Parameter         | Description                              |
|------------------|------------------------------------------|
| `data_root`       | Path to `avsbench_data/`                 |
| `checkpoint_path` | Path to the trained model checkpoint     |
| `output_dir`      | Directory to save evaluation results     |

---

## 📁 Repository Structure

```text
CAVS-CMR/
├── avs_scripts/
│   └── avss/
│       ├── train.py              # Training entry point
│       ├── test.py               # Evaluation entry point
│       ├── train.sh              # Training launch script
│       ├── test.sh               # Evaluation launch script
│       ├── config.py             # Configuration and path settings
│       ├── color_dataloader.py   # Dataset and label loading
│       ├── model/                # Model definitions (CMR, backbone, heads)
│       └── utils/                # Helper utilities
├── preprocess_scripts/           # Data preprocessing scripts
├── avsbench_data/                # AVSBench dataset (not included)
├── pretrained_backbones/         # Pretrained weights (not included)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📝 Notes

- This is a code-only release. Dataset folders, pretrained weights, checkpoints, logs, and paper source files are intentionally omitted.
- AVSBench code is included only as the benchmark foundation for AVSS experiments.
- Please refer to AVSBench for dataset download and license details.

---

## 📚 Citation

If you find this work useful, please consider citing CMR and AVSBench:

```bibtex
@article{your2026cmr,
  title     = {Continual Audio-Visual Segmentation},
  author    = {Your Name and Coauthors},
  journal   = {Transactions on Machine Learning Research},
  year      = {2026}
}

@inproceedings{zhou2022avsbench,
  title     = {Audio-Visual Segmentation},
  author    = {Zhou, Jinxing and others},
  booktitle = {ECCV},
  year      = {2022}
}
```

---

## License

This repository is released under the **Apache 2.0** license. See [LICENSE](LICENSE).
