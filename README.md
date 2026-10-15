# Ideological Noisy Channel Theory (INCT) — Empirical Replication Codebase

Official replication repository for:
> **Cognitive Entropy Minimization as a Driver of Ideological Rigidity**  
> *Target Venues:* Nature Human Behaviour / PNAS[cite: 1, 2]

[![Open In Colab: Study 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Open In Colab: Study 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository provides open-source replication scripts, feature-extraction pipelines, and empirical datasets for Paper 1 of the **Ideological Noisy Channel Theory (INCT)** trilogy[cite: 1, 2]. 

INCT models human cognition as a metabolically bounded communication channel constrained by channel capacity ($C_{\text{cog}}$)[cite: 1, 5]. When macro-environmental chaos and unpredictability ($H_{\text{env}}$) overwhelm this capacity, the biological prediction engine prevents metabolic exhaustion by adopting dogma ($R_{\text{int}} \to 1$) as a discrete, lossy error-correcting code[cite: 1, 5].

All empirical analyses in this repository are derived from **$N = 5,652$ standardized, length-normalized token windows** extracted from real-world historical treatises, political archives, and timestamped social network timelines.

---

## Repository Structure

```text
├── notebooks/
│   ├── Study_1_Cross_Sectional.ipynb        # Pan-Dogmatic cross-sectional NLP discovery (N = 1,679)
│   └── Study_2_Longitudinal_Network.ipynb    # Macro-historical & micro-network trajectory discovery (N = 3,973)
├── figures/
│   ├── Figure_1_Study1_CrossSectional.pdf    # Multi-panel cross-ideological distributions & weights
│   └── Figure_2_Study2_Longitudinal.pdf      # Crisis forcing & distributed network decay plots
├── src/
│   └── feature_extraction.py                 # Core length-invariant information-theoretic kernel
├── requirements.txt                          # Python library dependencies
└── README.md

---
