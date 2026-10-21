# Disentangling Closed-Class Binarization, Partisan Style, and Conversational Decompression in Ideological Discourse

Official replication repository for the manuscript:

> **Disentangling Closed-Class Binarization, Partisan Style, and Conversational Decompression in Ideological Discourse: A Five-Corpus Rate-Distortion Study**  
> **Author:** Kashif Shahzad  
> **arXiv Classification:** `cs.CL` (Computation and Language); `cs.SI` (Social and Information Networks)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository provides the complete, end-to-end Python replication pipeline, de-identified numeric feature matrices (`data/*.csv`), and dynamic vector figure generators (`figures/*.pdf`) for testing **Ideological Noisy Channel Theory (INCT)** across five benchmark text corpora ($N = 26,388$ fixed-length text windows).

Our pipeline separates **source-level closed-class quantization** (collapsing social actors and causal contingencies into collective intergroup pronouns _We/Us_ vs. _They/Them_ and deterministic absolutist quantifiers) from **channel-level open-class lexical compression** (Kullback-Leibler divergence $D_{\text{KL,Global}}$, Topic-Deconfounded Tail $D_{\text{KL,Tail}}$, Yule's $K$ concentration, and Lempel-Ziv compressibility $R_{\text{LZ}}$).

To eliminate common natural language processing (NLP) confounds, all analyses enforce:

1. **Strict Disjoint Vocabulary Filtering:** A 130-word closed-class stopword filter ($\mathcal{S}_{130}$) removes all function words, modals, negations, personal/collective pronouns, and absolutist quantifiers from the open-class content stream prior to computing $D_{\text{KL}}$, Yule's $K$, and $R_{\text{LZ}}$.
2. **Zero Multi-Author Concatenation & Fixed-Length Windowing ($W$):** Prevents Heaps' Law length distortion and cross-author "Frankenstein" windowing artifacts.
3. **Out-of-Author Cross-Validation (`GroupKFold`):** Groups train/test splits strictly by `user_id` with in-fold `StandardScaler` normalization.
4. **Speech-Level & Person-Mean Centered REML Mixed-Effects Models (`MixedLM`):** Controls for spoken vs. written delivery mode and secular drift in historical archives, and separates within-user conversational state shifts ($\Delta\phi_{it} = \phi_{it} - \bar{\phi}_i$) from between-user traits ($\bar{\phi}_i$) in social networks after stripping all retweets and `#hashtags`.

---

## Repository Structure

```text
├── data/                                                  # Exported de-identified feature matrices
│   ├── study1a_allsides_balanced_w52.csv                  # Study 1A: N=16,366 articles across 553 topics (W=52)
│   ├── study1b_single_posts_w40.csv                       # Study 1B: N=6,913 single posts, zero concatenation (W=40)
│   ├── study1c_stormfront_windows_w50.csv                 # Study 1C: 1,465 windows across N=728 unique users (W=50)
│   ├── study2a_sotu_speeches_1790_2008.csv                # Study 2A: N=222 SOTU speeches across 40 Presidents (W=250)
│   ├── study2b_isis_longitudinal_w150_no_hashtags.csv     # Study 2B: N=785 RT- & hashtag-stripped windows, 56 users (W=150)
│   └── study2b_sampling_audit_w100.csv                    # Study 2B Audit: N=207 users, 697 windows (W=100)
├── figures/
│   ├── Figure_1_RateDistortion_CrossSectional.pdf         # Vector PDF of Figure 1 (Studies 1B & 1C)
│   ├── Figure_1_RateDistortion_CrossSectional.png         # 300 DPI PNG preview of Figure 1
│   ├── Figure_2_Historical_And_Network_MixedLM.pdf        # Vector PDF of Figure 2 (Studies 2A & 2B)
│   └── Figure_2_Historical_And_Network_MixedLM.png        # 300 DPI PNG preview of Figure 2
├── notebooks/
│   └── Master_V3_Five_Corpus_Pipeline.ipynb               # Self-contained statistical audit & dynamic figure generator
├── src/
│   └── feature_extraction.py                              # 9-feature Rate-Distortion & Closed-Class Quantization kernel
├── requirements.txt                                       # Pinned Python dependencies
└── README.md
```
