# Does Ideological Pressure Compress Language? A Five-Corpus Robustness Audit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KashShahzad/inct-empirical-validation/blob/main/notebooks/Robustness_Reanalysis.ipynb)

Replication materials for Shahzad (2026), *Does Ideological Pressure Compress Language? A Five-Corpus Robustness Audit*.

## What the paper does

Text studies of ideology often report that extremist or crisis-era language uses a narrower vocabulary and more "us versus them" words. This project tests those claims on five public corpora with one feature pipeline and progressively stricter estimators: story matching, outlet- and author-level inference, random slopes, equivalence tests, and specification curves with joint tests.

In short: vocabulary-concentration effects are the most fragile and mostly disappear under appropriate clustering. Closed-class markers (plural pronouns, absolutist words) are more stable and behave like traits of the writer. See the paper for details.

| Study | Corpus | Window | Unit of inference |
|---|---|---|---|
| 1A | AllSides balanced news (Qbias) | 52 tokens | 443 outlets; 6,447 stories |
| 1B | Measuring Hate Speech | 40 tokens | 6,913 single posts |
| 1C | Stormfront | 50 tokens | 728 authors (61 with both hate and non-hate posts) |
| 2A | U.S. State of the Union, 1790–2008 | 250 tokens | 222 addresses; 40 presidents |
| 2B | Pro-ISIS Twitter timelines | 150 tokens | 56 accounts (785 windows) |
| 2B audit | Pro-ISIS vs. keyword-sampled tweets | 100 tokens | 207 accounts (697 windows) |

## Reproducing the results

Every table and figure in the paper is produced by one notebook, `notebooks/Robustness_Reanalysis.ipynb`. Open it in Colab with the badge above, or run it locally:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/Robustness_Reanalysis.ipynb
```

The notebook reads the feature matrices in `data/`, rebuilds the Study 1A windows from the public Qbias release, and writes everything to `robustness_outputs/`. Random seeds are fixed. The published outputs were produced with Python 3.13 and the package versions in `requirements.txt`; small differences in the last decimal place are possible with other versions, mainly in the cross-validated AUCs.

## Where each result comes from

| Paper | File in `robustness_outputs/` |
|---|---|
| Table 4, Table A1 | `study1a_outlet_level_results.csv` (outlet-text rows and title-included rows) |
| Table A2, Figure 1 | `study1a_leave_top_k_outlets.csv`, `Figure_R1_Study1A_outlet_level` |
| Table 5 | `study1b_table3.csv` |
| Table 6 | `study1c_table4.csv` |
| Table 7 | `study1c_classification_auc.csv` |
| Table 8 | `study2a_joint_test.csv` (all 288 fits per feature: `study2a_specification_curve.csv`) |
| Table 9 | `study2a_joint_test_by_crisis_family.csv` |
| Figures 2–4 | `Figure_R2_Study2A_speccurve_absolutist`, `Figure_R3_Study2A_speccurve_theythem`, `Figure_R4_Study2A_all_outcomes` |
| Table 10 | `study2b_robustness_main.csv` |
| Table 11 | `study2b_robustness_main.csv` (between-account columns), `study2b_joint_test.csv` (all 54 fits per feature: `study2b_specification_curve.csv`) |
| Figure 5 | `Figure_R5_Study2B_within_user` |
| Table 12 and the AUCs in Section 4.3 | `study2b_keyword_audit_table8.csv`, `study2b_keyword_audit_auc.csv` |

## Repository contents

```text
inct-empirical-validation/
├── notebooks/
│   ├── Robustness_Reanalysis.ipynb      # produces every table and figure in the paper
│   └── Master_V3_Five_Corpus_Pipeline.ipynb   # original baseline pipeline (see note below)
├── src/
│   └── feature_extraction.py            # the 11-feature kernel, identical to the notebook's
├── data/
│   ├── study1a_allsides_balanced_w52.csv          # Study 1A windows, AllSides title included (16,366)
│   ├── study1b_single_posts_w40.csv               # Study 1B single posts (6,913)
│   ├── study1c_stormfront_windows_w50.csv         # Study 1C windows (1,465) from 728 authors
│   ├── study2a_sotu_speeches_1790_2008.csv        # Study 2A address-level features (222)
│   ├── study2b_isis_longitudinal_w150_no_hashtags.csv  # Study 2B timeline windows (785)
│   └── study2b_sampling_audit_w100.csv            # Study 2B keyword-sample audit windows (697)
├── robustness_outputs/                  # all tables (CSV) and figures (PDF, PNG) used in the paper
├── figures/                             # figures from the original baseline analysis (not used in the paper)
├── requirements.txt
└── README.md
```

**About the baseline notebook and `figures/`.** `Master_V3_Five_Corpus_Pipeline.ipynb` and the figures in `figures/` come from the first version of this analysis. Several of its conclusions do not survive the checks in the paper, for example the crisis effect on vocabulary concentration in Study 2A. They are kept for transparency. The paper's results come only from `Robustness_Reanalysis.ipynb`.

**About the repository name.** "INCT" stands for Ideological Noisy Channel Theory, the working framework that motivated the project. The paper tests the linguistic predictions of that idea; it does not test the cognitive mechanism.

## Data sources and ethics

All corpora are public: AllSides balanced news via Qbias (Haak & Schaer, 2023), Measuring Hate Speech (Kennedy et al., 2020), the Stormfront hate-speech dataset (de Gibert et al., 2018), the Project Gutenberg State of the Union collection (#5050), and the Kaggle archives *How ISIS Uses Twitter* (fifthtribe) and *ISIS-related tweets* (activegalaxy). No private or deleted data were used, and no attempt was made to identify or contact any author. The Study 2B matrix keeps the public account handle from the Kaggle archive, only as a grouping variable for the models.

## Citation

Shahzad, K. (2026). *Does ideological pressure compress language? A five-corpus robustness audit*. https://github.com/KashShahzad/inct-empirical-validation
