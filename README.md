# Lexical Compression and Closed-Class Framing in Ideological Discourse: A Multi-Corpus Methodological and Specification-Curve Audit

[![Open Baseline Pipeline In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KashShahzad/inct-empirical-validation/blob/main/notebooks/Master_V3_Five_Corpus_Pipeline.ipynb)
[![Open Robustness & Spec-Curve Audit In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KashShahzad/inct-empirical-validation/blob/main/notebooks/Robustness_Reanalysis.ipynb)

Official replication repository for **"Lexical Compression and Closed-Class Framing in Ideological Discourse: A Multi-Corpus Methodological and Specification-Curve Audit"** (Shahzad, 2026).

---

## Overview

Computational studies of political and extremist text frequently report that ideological rigidity is characterized by reduced open-class vocabulary diversity and elevated closed-class binary framing. This repository provides an end-to-end methodological and specification-curve (multiverse) audit across five political and extremist corpora, isolating substantive linguistic signal from seven pervasive corpus confounds:

1. Cross-document concatenation and Heaps'-law length scaling
2. Function-word leakage into open-class content streams
3. Platform retweets, hashtag templates, and aggregator roundup titles
4. Keyword-scraped control baselines (`AboutIsis.csv`)
5. Organizational outlet clustering vs. within-story matching (_AllSides_)
6. Secular historical modernization drift vs. war/recession crisis taxonomies (_State of the Union_, 1790–2008)
7. Within- vs. between-author conflation and omitted random slopes (_Stormfront_ & _ISIS_ Twitter)

---

## Repository Structure

```text
inct-empirical-validation/
├── notebooks/
│   ├── Master_V3_Five_Corpus_Pipeline.ipynb          # Primary feature extraction & baseline tables (Studies 1A–2B)
│   └── Robustness_Reanalysis.ipynb                   # Outlet-level 1A, 288-spec curve 2A, & within-user 2B reanalysis
├── data/                                             # Exported window- and unit-level feature matrices (CSV)
│   ├── study1a_allsides_balanced_w52.csv             # Study 1A baseline windows (N = 16,366)
│   ├── study1b_mhs_single_posts_w40.csv              # Study 1B single social media posts (N = 6,913)
│   ├── study1c_stormfront_users_w50.csv              # Study 1C unique Stormfront authors (N = 728; 61 dual-label)
│   ├── study2a_sotu_speeches_1790_2008.csv           # Study 2A speech-level SOTU matrix (N = 222 speeches, 40 Presidents)
│   ├── study2b_isis_longitudinal_w150_no_hashtags.csv# Study 2B chronological RT/hashtag-stripped panel (N = 785 windows)
│   └── study2b_table6_keyword_scrape_audit_w100.csv  # Table 8 keyword-scraped baseline audit (N = 207 accounts)
├── robustness_outputs/                               # Multi-estimator & specification-curve outputs and figures
│   ├── study1a_outlet_level_results.csv              # 4-estimator comparison (outlet-only n=15,038 & title-included n=16,366)
│   ├── study1a_leave_top_k_outlets.csv               # Leave-top-k (k=0..10) within-story vs. unmatched sensitivity
│   ├── study2a_specification_curve.csv               # Full 288-specification curve per feature (2,592 fits)
│   ├── study2a_joint_test.csv                        # Circular-shift permutation joint test (203 rotations)
│   ├── study2a_joint_test_by_crisis_family.csv       # Joint tests stratified by War vs. Economic crisis families
│   ├── study2b_robustness_main.csv                   # User RI, User RI + random slope, User FE (wild bootstrap), cap20, drop3
│   ├── study2b_specification_curve.csv               # 54-specification curve across Study 2B panel models
│   ├── study2b_joint_test.csv                        # Within-user permutation joint test for Study 2B
│   ├── Figure_R1_Study1A_outlet_level.pdf/.png       # Leave-top-k curves & two-way FE outlet effects
│   ├── Figure_R2_Study2A_speccurve_absolutist.pdf/.png # 288-spec curve for absolutist density
│   ├── Figure_R3_Study2A_speccurve_theythem.pdf/.png # 288-spec curve for they/them pronoun density
│   ├── Figure_R4_Study2A_all_outcomes.pdf/.png       # Summary specification curves across all 9 SOTU outcomes
│   └── Figure_R5_Study2B_within_user.pdf/.png        # Study 2B within-account estimators & 54-spec joint test summary
├── figures/                                          # Baseline multi-panel figures
│   ├── Figure_1_CrossSectional_AllSides_MHS_Stormfront.pdf/.png
│   └── Figure_2_Historical_And_Network_MixedLM.pdf/.png
├── src/
│   └── feature_extraction.py                         # Standardized 11-feature kernel (synced verbatim with notebooks)
├── requirements.txt                                  # Pinned Python dependencies
└── README.md
```
