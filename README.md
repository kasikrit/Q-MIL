# Quantum-Inspired Multiple Instance Learning (Q-MIL) for Epistemic Triage in Hematopathology

This repository contains the source code for the hybrid AI framework that integrates quantum state embedding, Hamiltonian attention, and von Neumann entropy-driven double-verification triage for the clinical screening of Iron Deficiency Anemia (IDA) and Thalassemia (THL).

**Paper Title:** Quantum-Inspired Entanglement and Uncertainty Quantification for Strict Double-Verification Triage in Clinical Hematopathology

**Journal**: Scientific Reports (Submitted)

**Citation**: *Pending*

**Abstract:**

Multiple Instance Learning (MIL) applied at the patient level is inherently susceptible to overfitting and noise. In binary classification tasks, this vulnerability is often exacerbated by biases stemming from dataset composition and experimental design. When predictive probabilities fall into a zone of uncertainty, forcing a model to make a definitive classification is clinically impractical; it compromises the diagnostic reliability required for treatment planning and drives up the costs of unnecessary confirmatory testing. To address this, we proposes a novel Quantum-inspired MIL (Q-MIL) architecture that embeds segmented red blood cells (RBCs) as entangled quantum states. A double-verification triage framework, driven by von Neumann entropy, is implemented to prevent hazardous forced predictions in highly ambiguous cases. Furthermore, a native Explainable AI (XAI) suite is designed to eliminate the need for post-hoc attribution by allowing direct inspection of the wave function collapse mechanism itself. Rigorous statistical analyses demonstrate that the triage mechanism successfully isolates potential model failures, achieving a statistically significant accuracy disparity between the confident autonomous track and the routed expert-review track ($$91.1\%$$ vs. $$68.2\%$$, $$p < 0.001$$). Ultimately, this work delivers the critical diagnostic safety gate that universal AI models currently lack.

## Hardware and Software Specifications

* **Python Version:** 3.9.16
* **TensorFlow Version:** 2.11.0 *(Utilized for CNN backbone feature extraction)*
* **PyTorch Version:** 1.13.0 *(Utilized for Q-MIL head, quantum latent projections, and gradient computation)*

## Creating a Python Environment

We recommend using Anaconda or Miniconda to manage the environment.

```bash
conda create -n qmil_env python=3.9.16
conda activate qmil_env
```

## Installing Required Software Packages

Ensure to install the correct PyTorch wheels for your specific CUDA version (e.g., CUDA 11.7 for PyTorch 1.13).

```bash
pip install tensorflow==2.11.0
pip install torch==1.13.0 torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn yacs tqdm opencv-python Pillow
```

## Available Dataset

The datasets generated and/or analyzed during the current study are not publicly available as they are being utilized for ongoing and future research, but are available from the corresponding author on reasonable request. However, a related dataset with similar attributes is publicly accessible at [https://github.com/kasikrit/IDA-THL-Classification/](https://github.com/kasikrit/IDA-THL-Classification/).

## Q-MIL Pipeline Execution

### 1. Configure Pipeline Parameters
Modify the architectural parameters, learning rates, and directory paths via the central `config.py` file utilizing `yacs`.

```python
# config.py
import os
from yacs.config import CfgNode as CN

_C = CN()

# ===== System & Path Configuration =====
_C.SYSTEM = CN()
_C.SYSTEM.SEED = 1337                  # Enforces strict reproducibility
_C.SYSTEM.BASE_DIR = "./"
_C.SYSTEM.DATA_DIR = os.path.join(_C.SYSTEM.BASE_DIR, "data")
_C.SYSTEM.WEIGHTS_DIR = os.path.join(_C.SYSTEM.BASE_DIR, "models")
_C.SYSTEM.OUTPUT_DIR = os.path.join(_C.SYSTEM.BASE_DIR, "results")

# ===== Model Architecture =====
_C.MODEL = CN()
_C.MODEL.BACKBONE_NAME = "ConvNeXtLarge"
_C.MODEL.D_IN = 1536                   # Feature dimension from ConvNeXt-Large
_C.MODEL.D_MODEL = 256                 # Quantum latent projection dimension
_C.MODEL.NUM_CLASSES = 2               # Binary classification (IDA vs. THL)

# ===== Training Configuration =====
_C.TRAIN = CN()
_C.TRAIN.FOLDS = [0, 1, 2, 3, 4]       # Active folds for Monte Carlo ensemble
_C.TRAIN.EPOCHS = 60
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR_BACKBONE = 1e-5
_C.TRAIN.LR_QMIL = 1e-4
```

### 2. Start Monte Carlo Model Training
Execute the training phase to extract classical CNN features, project them into the Hilbert space, and train the independent Q-MIL ensemble folds.

```bash
python main.py --mode train --data_dir ./data/ --weights_dir ./models/
```

## Patient-Level Classification: Epistemic Triage

### 1. Run Threshold Calibration (Phase 1B)
Executes Out-of-Fold (OOF) evaluation to establish the strict epistemic entropy boundary ($$H_{limit}$$) on unseen validation patients, entirely preventing test-set data leakage.

```bash
python main.py --mode calibrate
```

### 2. Independent Test Inference (Phase 3)
Evaluates the soft-voting ensemble on the held-out test set and enforces the double-verification clinical triage gate based on the calibrated $$H_{limit}$$.

```bash
python main.py --mode test
```

### 3. Generate Explainable AI Visualizations (Phase 4)
Produces the 3-panel XAI suite for a targeted patient, directly rendering the mathematical etiology of the von Neumann entropy.
* **Panel 1:** Morphological Latent Space (2D PCA)
* **Panel 2:** Quantum Mixed State ($$\rho$$)
* **Panel 3:** Antagonistic Feature Activation ($$\rho \odot O_{net}$$)

```bash
python main.py --mode xai
```

## Output
Execution of the phases above will automatically compile and save the evaluation metrics, triage performance, density matrices, and `Test_Ensemble_Predictions.csv` in the designated `./results/` directory.
