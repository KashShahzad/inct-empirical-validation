# Ideological Noisy Channel Theory (INCT) — Empirical Replication Codebase

Official replication repository for:
> **Cognitive Entropy Minimization as a Driver of Ideological Rigidity: A Rate-Distortion and Noisy-Channel Framework**  
> **Author:** Kashif Shahzad  
> *Target Venues:* Nature Human Behaviour / PNAS

[![Open In Colab: Deconfounded Empirical Pipeline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository contains the complete, self-contained replication scripts, deconfounded feature-extraction kernel, and publication figure generators for **Ideological Noisy Channel Theory (INCT)**.

INCT models human cognition as a metabolically bounded information-processing channel constrained by finite capacity ($C_{\text{cog}}$). When macro-environmental volatility ($H_{\text{env}}(t)$) or unbuffered predictive friction threatens to exceed channel capacity, the cognitive system minimizes variational free energy via rate-distortion compression: contracting its open-class lexical probability distribution ($D_{\text{KL}}$, Yule's $K$), collapsing continuous social categories into binary *Us-versus-Them* and absolutist codes, or increasing sequential Lempel-Ziv redundancy ($R_{\text{LZ}}$).

All empirical analyses enforce strict fixed-length token windowing ($W$), zero cross-author concatenation, explicit retweet stripping, out-of-user `GroupKFold` cross-validation, and detrended Linear Mixed-Effects Models (`MixedLM`).

---

## Repository Structure

```text
├── notebooks/
│   ├── Run_1_And_2_Deconfounded_Audit.ipynb          # Complete statistical pipeline (Studies 1A, 1B, 2A, 2B)
│   └── Generate_Publication_Figures.ipynb            # Self-contained 300 DPI PDF/PNG figure generator
├── figures/
│   ├── Figure_1_RateDistortion_CrossSectional.pdf    # Study 1A dose-response & Study 1B Stormfront GroupKFold
│   ├── Figure_1_RateDistortion_CrossSectional.png    # 300 DPI raster preview of Figure 1
│   ├── Figure_2_Historical_And_Network_MixedLM.pdf   # Study 2A SOTU crisis MixedLM & Study 2B network coupling
│   └── Figure_2_Historical_And_Network_MixedLM.png   # 300 DPI raster preview of Figure 2
├── src/
│   └── feature_extraction.py                         # 7-feature Rate-Distortion & binarization kernel
├── requirements.txt                                  # Pinned Python dependencies (including statsmodels)
└── README.md