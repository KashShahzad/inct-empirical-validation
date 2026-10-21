"""
Ideological Noisy Channel Theory (INCT) — V3 Feature Extraction Kernel
======================================================================
Implements the 130-word disjoint closed-class stopword filter, retweet and
hashtag stripping utilities, and the 9-dimensional Rate-Distortion and
Closed-Class Quantization feature kernel used across Studies 1A, 1B, 1C, 2A, and 2B.
"""

import math
import re
import zlib
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np

# 1. Collective Intergroup Pronoun Lexicons (Us-vs-Them Partitioning)
WE_US_PRONOUNS = {"we", "us", "our", "ours", "ourselves"}
THEY_THEM_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}
US_THEM_PRONOUNS = WE_US_PRONOUNS | THEY_THEM_PRONOUNS

# 2. Validated 19-Word Absolutist Lexicon (Al-Mosaiwi & Johnstone, 2018)
ABSOLUTIST_LEXICON = {
    "absolutely", "all", "always", "complete", "completely", "constant", "constantly",
    "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
    "never", "nothing", "totally", "whole"
}

# 3. Disjoint 130-Word Closed-Class Stopword Set
# Strips all function words, auxiliaries, modals, negations, personal/collective pronouns,
# and absolutist quantifiers from the open-class content stream (X_c).
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

FEATURE_COLS = [
    "D_KL_Global",
    "D_KL_Tail",
    "Yules_K",
    "R_LZ_Content",
    "Bigram_Redundancy",
    "Absolutist_Density",
    "UsThem_Pronoun_Density",
    "We_Us_Density",
    "They_Them_Density"
]


def is_retweet_or_quote(text: str) -> bool:
    """
    Identifies verbatim retweets, quote-tweets, and viral forwards so they can be
    excluded prior to chronological author-level windowing.
    """
    t = str(text).strip().lower()
    if t.startswith("rt ") or t.startswith("rt:") or " rt @" in t or "via @" in t or "♻" in t:
        return True
    return bool(re.search(r"\brt\s*@\w+", t) or re.search(r"^@\w+\s*:", t))


def clean_and_tokenize(text: str, strip_hashtags: bool = True) -> List[str]:
    """
    Strips URLs, @mentions, and optional #hashtags, returning lowercase
    alphabetical tokens of length >= 2.
    """
    pattern = r"http\S+|@\S+|#\S+|\brt\b" if strip_hashtags else r"http\S+|@\S+|\brt\b"
    cleaned = re.sub(pattern, "", str(text).lower())
    return re.findall(r"\b[a-z]{2,}\b", cleaned)


def extract_rate_distortion_features(
    tokens: Sequence[str],
    min_content: int = 12
) -> Optional[Dict[str, float]]:
    """
    Computes the 9 deconfounded Rate-Distortion and Closed-Class Quantization
    metrics on a fixed-length W-token window.
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

    # 1. Global Content KL Divergence from Uniformity: D_KL = log2(V_c) - H_c
    H_c = -sum((c / N_c) * math.log2(c / N_c) for c in counts_c.values())
    D_KL_global = math.log2(V_c) - H_c

    # 2. Topic-Deconfounded Tail KL Divergence (stripping top-3 content types)
    top3 = {w for w, _ in counts_c.most_common(3)}
    tail_tokens = [t for t in content_tokens if t not in top3]
    counts_tail = Counter(tail_tokens)
    N_tail, V_tail = float(len(tail_tokens)), len(counts_tail)
    if V_tail > 2 and N_tail > 5:
        H_tail = -sum((c / N_tail) * math.log2(c / N_tail) for c in counts_tail.values())
        D_KL_tail = math.log2(V_tail) - H_tail
    else:
        D_KL_tail = 0.0

    # 3. Yule's K Lexical Collision Concentration
    freq_spec = Counter(counts_c.values())
    M2 = sum((m ** 2) * v_m for m, v_m in freq_spec.items())
    yules_K = 10000.0 * (M2 - N_c) / (N_c ** 2)

    # 4. Content Lempel-Ziv Algorithmic Compressibility (R_LZ)
    raw_bytes = " ".join(content_tokens).encode("utf-8")
    R_lz = 1.0 - (len(zlib.compress(raw_bytes, level=9)) / float(len(raw_bytes)))

    # 5. Contiguous Bigram Redundancy
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bg_red = 1.0 - (len(set(bigrams)) / float(len(bigrams))) if bigrams else 0.0

    # 6-9. Closed-Class Quantization Densities
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
        "Absolutist_Density": abs_dens,
        "UsThem_Pronoun_Density": we_dens + they_dens,
        "We_Us_Density": we_dens,
        "They_Them_Density": they_dens
    }


def cohens_d(x0: Iterable[float], x1: Iterable[float]) -> float:
    """Computes pooled-variance standardized Cohen's d (x1 - x0)."""
    arr0 = np.asarray(list(x0), dtype=float)
    arr1 = np.asarray(list(x1), dtype=float)
    s0, s1 = np.std(arr0, ddof=1), np.std(arr1, ddof=1)
    n0, n1 = len(arr0), len(arr1)
    pooled = np.sqrt(((n0 - 1) * (s0 ** 2) + (n1 - 1) * (s1 ** 2)) / max(1, (n0 + n1 - 2)))
    return float((np.mean(arr1) - np.mean(arr0)) / (pooled + 1e-12))


if __name__ == "__main__":
    sample = (
        "We must always defend our community completely because they will never stop "
        "attacking our people and our traditions. Every citizen must stand united "
        "against their total hostility, for we are forever committed to our cause."
    )
    toks = clean_and_tokenize(sample, strip_hashtags=True)
    res = extract_rate_distortion_features(toks, min_content=10)
    print(f"Extracted {len(toks)} tokens | Disjoint V3 Feature Output:")
    for k, v in (res or {}).items():
        print(f"  {k:<24}: {v:.4f}")