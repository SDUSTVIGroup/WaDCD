# ✨ WaDCD✨

**A PyTorch Implementation for Multi-Class Unsupervised Anomaly Detection**

> **WaDCD: Wavelet-Aware Deviation-Correction Diffusion for Multi-Class Unsupervised Anomaly Detection**

---

## 🎨 Approach

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

- [MVTec-AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)

After downloading, organize the dataset and specify its path using `--data-dir`.

---

## 🏋️ Training

Train WaDCD with the following command:

```
torchrun train_WaDCD.py \
    --dataset mvtec \
    --data-dir /path/to/dataset \
    --model-size UNet_L \
    --object-category all \
    --image-size 288 \
    --center-size 256 \
    --center-crop True \
    --wavelet-loss True \
    --dual-schedule True \
    --noise-schedule-low squaredcos_cap_v2 \
    --noise-schedule-high linear \
    --auto-lambda-hf true \
    --use-wavelet-sdem-gate true \
    --wavelet-combine-mode sum
```
## 🧪 Testing

After training, evaluate the model using:

```
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
    --dual-schedule True \
    --noise-schedule-low squaredcos_cap_v2 \
    --noise-schedule-high linear \
    --dual-schedule-sampling true \
    --use-wavelet-sdem-gate true \
    --wavelet-combine-mode sum
```

---

## 📊 Results

The performance of WaDCD on MVTec-AD and VisA is reported in the table below.

The following results are obtained from our best trained models. Since our experiments were primarily conducted on the **MVTec-AD** dataset, we provide the corresponding pretrained weights for **WaDCD UNet_L** to facilitate evaluation and reproducibility. The pretrained weights can be downloaded from [here](https://github.com/SDUSTVIGroup/WaDCD/releases/download/v1.0/WaDCD_mvtec_UNet_L.pt).

| Dataset | I-AUROC | I-AP | I-f1max | P-AUROC | P-AP | P-f1max | P-AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| MVTec-AD | 99.1 | 99.7 | 98.5 | 98.9 | 77.4 | 71.9 | 94.5 |
| VisA | 96.0 | 96.8 | 92.3 | 98.5 | 53.0 | 51.9 | 91.3 |

---

## 📸 Sample Results

Below are some sample outputs showing the performance of WaDCD on real anomaly detection data.

<img width="1517" height="857" alt="Fig 3" src="https://github.com/user-attachments/assets/3195d267-5f46-4721-8d3e-aa6a3a0f73bc" />
<img width="1516" height="853" alt="Fig 4" src="https://github.com/user-attachments/assets/6f700767-8064-469b-baa3-4c3654e90bde" />

---

## 📚 Citation

If you find this project useful, please consider starring this repository.

Citation information will be updated once the paper is available.
