"""
src/feature_extraction.py
=========================
Standardized 11-feature lexical compression and closed-class framing kernel.
Synchronized verbatim with Master_V3_Five_Corpus_Pipeline.ipynb and
Robustness_Reanalysis.ipynb so all exported CSVs and tables match 100%.
"""

import math
import re
import zlib
from collections import Counter
import numpy as np
import pandas as pd

WE_US_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
THEY_THEM_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}
US_THEM_PRONOUNS = WE_US_PRONOUNS | THEY_THEM_PRONOUNS

ABSOLUTIST_LEXICON = {
    "absolutely", "all", "always", "complete", "completely", "constant", "constantly",
    "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
    "never", "nothing", "totally", "whole"
}

STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "as", "with", "on",
    "was", "be", "by", "at", "this", "are", "from", "or", "an", "have", "had", "has",
    "but", "which", "one", "would", "were", "there", "been", "if", "more", "when", "so",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "me", "my", "mine", "myself",
    "not", "no", "nor", "do", "does", "did", "doing", "can", "could", "should", "will",
    "shall", "may", "might", "who", "whom", "whose", "what", "where", "why", "how",
    "about", "into", "through", "during", "before", "after", "above", "below", "between",
    "under", "again", "further", "then", "once", "here", "any", "both", "each", "few",
    "other", "some", "such", "only", "own", "same", "than", "too", "very", "just", "now"
} | US_THEM_PRONOUNS | ABSOLUTIST_LEXICON


def tokenize_clean(text: str, strip_hashtags: bool = False):
    """Tokenizes raw text into lowercase alphabetical words (>=2 chars) after stripping URLs/@handles."""
    pattern = r"http\S+|@\S+|#\S+" if strip_hashtags else r"http\S+|@\S+"
    cleaned = re.sub(pattern, "", str(text).lower())
    return re.findall(r"\b[a-z]{2,}\b", cleaned)


def extract_rate_distortion_features(tokens, min_content: int = 12):
    """
    Computes the 11 open-class compression and closed-class framing features
    on a fixed-length token window `tokens`.
    Matches the notebook kernel in Robustness_Reanalysis.ipynb verbatim.
    """
    if not tokens:
        return None
    W = float(len(tokens))
    content_tokens = [t for t in tokens if t not in STOPWORDS]
    N_c = float(len(content_tokens))
    if N_c < min_content:
        return None
    counts_c = Counter(content_tokens)
    V_c = len(counts_c)
    if V_c <= 3:
        return None

    # 1-2. Content Shannon entropy and Global KL divergence from uniform
    H_c = -sum((c / N_c) * math.log2(c / N_c) for c in counts_c.values())
    D_KL_global = math.log2(V_c) - H_c

    # 3. Top-3-deconfounded Tail KL divergence
    top3 = {w for w, _ in counts_c.most_common(3)}
    tail_tokens = [t for t in content_tokens if t not in top3]
    counts_tail = Counter(tail_tokens)
    N_tail, V_tail = float(len(tail_tokens)), len(counts_tail)
    if V_tail > 2 and N_tail > 5:
        H_tail = -sum((c / N_tail) * math.log2(c / N_tail) for c in counts_tail.values())
        D_KL_tail = math.log2(V_tail) - H_tail
    else:
        D_KL_tail = 0.0

    # 4. Yule's K repeat concentration
    freq_spec = Counter(counts_c.values())
    M2 = sum((m ** 2) * v_m for m, v_m in freq_spec.items())
    yules_K = 10000.0 * (M2 - N_c) / (N_c ** 2)

    # 5. Content Lempel-Ziv compressibility (zlib level 9)
    raw_bytes = " ".join(content_tokens).encode("utf-8")
    R_lz = 1.0 - (len(zlib.compress(raw_bytes, level=9)) / float(len(raw_bytes)))

    # 6-7. Contiguous bigram redundancy and conditional bigram entropy
    counts_all = Counter(tokens)
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bg_red = 1.0 - (len(set(bigrams)) / float(len(bigrams))) if bigrams else 0.0
    bg_counts = Counter(bigrams)
    H_cond = (
        -sum((c / len(bigrams)) * math.log2(c / counts_all[bg[0]]) for bg, c in bg_counts.items())
        if bigrams else 0.0
    )

    # 8-11. Closed-class marker densities
    abs_dens = sum(1 for t in tokens if t in ABSOLUTIST_LEXICON) / W
    we_dens = sum(1 for t in tokens if t in WE_US_PRONOUNS) / W
    they_dens = sum(1 for t in tokens if t in THEY_THEM_PRONOUNS) / W

    return {
        "H_Content": H_c,
        "D_KL_Global": D_KL_global,
        "D_KL_Tail": D_KL_tail,
        "Yules_K": yules_K,
        "R_LZ_Content": R_lz,
        "Bigram_Redundancy": bg_red,
        "H_Conditional": H_cond,
        "Absolutist_Density": abs_dens,
        "UsThem_Pronoun_Density": we_dens + they_dens,
        "We_Us_Density": we_dens,
        "They_Them_Density": they_dens,
    }


def cohens_d(x0, x1) -> float:
    """Pooled-SD Cohen's d (x1 minus x0)."""
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    s0, s1 = np.std(x0, ddof=1), np.std(x1, ddof=1)
    n0, n1 = len(x0), len(x1)
    pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / max(1, (n0 + n1 - 2)))
    return float((np.mean(x1) - np.mean(x0)) / (pooled + 1e-12))


def zscore(s):
    """Standardizes a pandas Series to zero mean and unit variance."""
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / s.std()