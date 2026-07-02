# Continual Audio-Visual Segmentation

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

This repository provides the official code release for our continual audio-visual segmentation setting. The core method is CMR, a rehearsal-based framework that tackles multi-modal semantic drift and co-occurrence confusion through multi-modal sample selection and collision-based sample rehearsal.

The implementation is built on the AVSBench benchmark codebase and focuses on the AVSS setting in `avs_scripts/avss/`.

## 🪵 TODO List

- ✅ Release core AVSS codebase
- ✅ Organize README as a project homepage
- ✅ Document data and checkpoint paths
- ⏳ Add more detailed usage notes if needed

## 🔥 What's New

- **(2026.7.2)** The README has been reorganized into a template-style project homepage.
- **(2026.7.2)** The repository now emphasizes the AVSS codebase, data preparation, and run commands.

# 🧠 CMR: Continual Audio-Visual Segmentation

> Official code release for the continual audio-visual segmentation setting.

This workspace contains the code for AVSS experiments and supporting AVSBench components. It is intended for running the training and evaluation pipeline, preparing data, and reproducing the continual audio-visual segmentation setting used by the project.

---

## 📌 Overview

Continual audio-visual segmentation extends AVSBench to a sequential learning setting. The code in this repository centers on the AVSS pipeline, including data loading, model definition, training, testing, and preprocessing utilities.

### What Is Included

- `avs_scripts/avss/train.py`: training entry point
- `avs_scripts/avss/test.py`: evaluation entry point
- `avs_scripts/avss/color_dataloader.py`: dataset and label loading
- `avs_scripts/avss/model/`: model definitions
- `avs_scripts/avss/utils/`: helper utilities
- `preprocess_scripts/`: preprocessing scripts for preparing data

---

## 🚀 Get Started

```bash
cd avs_scripts/avss
bash train.sh
bash test.sh
```

### Data Preparation

Place the AVSBench data under `avsbench_data` and the pretrained backbones under `pretrained_backbones`.

Update the data and backbone paths in `avs_scripts/avss/config.py` before running experiments.

### Notes

- This is a code-only release. Dataset folders, pretrained weights, checkpoints, logs, and paper source files are intentionally omitted from the public upload.
- The AVSBench code is included only as the benchmark foundation for the AVSS experiments.

## 📚 Citation

If you use this codebase, please cite the corresponding work and benchmark papers used by the project.

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
