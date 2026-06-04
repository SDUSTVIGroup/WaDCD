# ✨ WaDCD✨

**A PyTorch Implementation for Multi-Class Unsupervised Anomaly Detection**

> **WaDCD: Wavelet-Aware Deviation-Correction Diffusion for Multi-Class Unsupervised Anomaly Detection**

---

## 🎨 Approach

**WaDCD** is a reformulated diffusion-based framework designed for multi-class unsupervised anomaly detection.  
It learns the distribution of normal samples and detects anomalies by correcting deviations from normality.
<img width="1767" height="845" alt="image1" src="https://github.com/user-attachments/assets/2613fac5-24a7-4f0c-96c4-57b891d60dbc" />
<img width="1764" height="892" alt="image2" src="https://github.com/user-attachments/assets/44d629d1-83e9-4f0e-afe8-8585acbcdc87" />


---

## 🚀 Getting Started

### 🛠️ Environment Setup

We use **Python 3.11** for all experiments.

Install the required packages with:

```bash
pip3 install -r requirements.txt
```

---

## 📁 Datasets

Please download the datasets from the official sources:

- [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff)

After downloading, organize the dataset and specify its path using `--data-dir`.

---

## 🏋️ Training

Train WaDCD with the following command:

```bash
torchrun train_WaDCD.py \
    --dataset mvtec \
    --data-dir /path/to/dataset \
    --model-size UNet_L \
    --object-category all \
    --image-size 288 \
    --center-size 256 \
    --center-crop True
```

### Arguments

| Argument | Description |
|---|---|
| `--dataset` | Dataset name, choose `mvtec` or `visa` |
| `--data-dir` | Path to the dataset |
| `--model-size` | Model size, e.g. `UNet_L` |
| `--object-category` | Object category to train on, use `all` for multi-class training |
| `--image-size` | Input image size |
| `--center-size` | Center crop size |
| `--center-crop` | Whether to apply center crop |

---

## 🧪 Testing

After training, evaluate the model using:

```bash
python evaluation_WaDCD.py \
    --dataset mvtec \
    --data-dir /path/to/dataset \
    --model-size UNet_L \
    --object-category all \
    --anomaly-class all \
    --image-size 288 \
    --center-size 256 \
    --center-crop True \
    --model-path /path/to/pretrained_weights.pt
```

---

## 📦 Pretrained Weights

We provide pretrained weights for **WaDCD UNet_L** for rapid inference and further experimentation.

### MVTec-AD Pretrained Weights

- [Download from Google Drive](#)

### VisA Pretrained Weights

- [Download from Google Drive](#)

---

## 🔥 ImageNet Pretrained Model

Using an ImageNet-pretrained model can slightly improve performance and robustness.

You can download the pretrained model provided by the LDM repository:

- [Download LDM Pretrained Model](#)

To use the pretrained model, set:

```bash
--from-scratch False \
--pretrained /path/to/pretrained_checkpoint.ckpt
```

---

## 📊 Results

The following results are obtained using the best trained models provided in this repository.  
They may differ slightly from the results reported in the paper, as they were obtained using optimized training parameters.

| Dataset | I-AUROC | I-AP | I-f1max | P-AUROC | P-AP | P-f1max | P-AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| MVTec-AD | 99.1 | 99.7 | 98.5 | 98.9 | 77.4 | 71.9 | 94.5 |
| VisA | 96.0 | 96.8 | 92.3 | 98.5 | 53.0 | 51.9 | 91.3 |

---

## 📸 Sample Results

Below are some sample outputs showing the performance of WaDCD on real anomaly detection data.

![WaDCD Samples](assets/sample_results.png)

> Please replace `assets/sample_results.png` with the actual path to your sample result image.

---

## 📚 Citation

If you find WaDCD useful in your research, please cite our work:

```bibtex
@inproceedings{beizaee2025correcting,
  title={Correcting deviations from normality: A reformulated diffusion model for multi-class unsupervised anomaly detection},
  author={Beizaee, Farzad and Lodygensky, Gregory A and Desrosiers, Christian and Dolz, Jose},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19088--19097},
  year={2025}
}
```

---

## 📄 License

Please refer to the repository license for details.

---

## 🙏 Acknowledgements

This project builds upon diffusion-based generative modeling and unsupervised anomaly detection research.  
We thank the authors of MVTec-AD, VisA, and LDM for their valuable datasets and pretrained models.
