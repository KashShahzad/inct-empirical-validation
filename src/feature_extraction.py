"""
Ideological Noisy Channel Theory (INCT) — Core Feature Extraction Module
========================================================================
Extracts the 7 length-invariant Rate-Distortion and Categorical Binarization
features across fixed-length W-token windows, alongside retweet filtering
and standardized effect-size utilities used in Studies 1A, 1B, 2A, and 2B.
"""

import math
import re
import zlib
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np

# 1. Closed-Class Syntactic Stopwords (35 function words)
STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "that", "it", "for", "as", "with", "on",
    "was", "be", "by", "at", "this", "are", "from", "or", "an", "have", "had", "has",
    "but", "which", "one", "would", "were", "there", "been", "if", "more", "when", "so"
}

# 2. Validated 19-Word Absolutist Lexicon (Al-Mosaiwi & Johnstone, 2018)
ABSOLUTIST_LEXICON = {
    "absolutely", "all", "always", "complete", "completely", "constant", "constantly",
    "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
    "never", "nothing", "totally", "whole"
}

# 3. Closed-Class Collective Intergroup Pronouns (Us-vs-Them Binarization)
US_THEM_PRONOUNS = {
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves"
}

# 4. Canonical Feature Column Order
FEATURE_COLS = [
    "D_KL_Global",
    "D_KL_Tail",
    "Yules_K",
    "R_LZ_Content",
    "Bigram_Redundancy",
    "Absolutist_Density",
    "UsThem_Pronoun_Density"
]


def is_retweet_or_quote(text: str) -> bool:
    """
    Identifies verbatim retweets, quote-tweets, and viral forwards so they can be
    stripped prior to author-level psycholinguistic windowing (Study 2B).
    """
    t = str(text).strip().lower()
    if t.startswith("rt ") or t.startswith("rt:") or " rt @" in t or "via @" in t or "♻" in t:
        return True
    return bool(re.search(r"\brt\s*@\w+", t) or re.search(r"^@\w+\s*:", t))


def clean_and_tokenize(text: str, strip_hashtags: bool = False) -> List[str]:
    """
    Strips URLs, user handles (@mentions), and optional hashtags, returning
    lowercase alphabetical tokens of length >= 2.
    """
    pattern = r"http\S+|@\S+|#\S+" if strip_hashtags else r"http\S+|@\S+"
    cleaned = re.sub(pattern, "", str(text).lower())
    return re.findall(r"\b[a-z]{2,}\b", cleaned)


def extract_rate_distortion_features(
    tokens: Sequence[str],
    min_content: int = 15
) -> Optional[Dict[str, float]]:
    """
    Computes the 7 deconfounded Rate-Distortion and Categorical Binarization
    metrics on a fixed-length W-token window.

    Parameters
    ----------
    tokens : Sequence[str]
        A fixed-length list/slice of W lowercase alphabetical word tokens.
    min_content : int, default=15
        Minimum required open-class content tokens after stopword removal.

    Returns
    -------
    dict or None
        Dictionary containing the 7 features in FEATURE_COLS, or None if the
        window contains fewer than `min_content` substantive tokens.
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

    # 2. Topic-Deconfounded Tail KL Divergence (removing top-3 frequent content words)
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

    # 6. Validated Absolutist Lexical Density (Al-Mosaiwi & Johnstone, 2018)
    abs_dens = sum(1 for t in tokens if t in ABSOLUTIST_LEXICON) / W

    # 7. Us-vs-Them Intergroup Pronoun Density
    usthem_dens = sum(1 for t in tokens if t in US_THEM_PRONOUNS) / W

    return {
        "D_KL_Global": D_KL_global,
        "D_KL_Tail": D_KL_tail,
        "Yules_K": yules_K,
        "R_LZ_Content": R_lz,
        "Bigram_Redundancy": bg_red,
        "Absolutist_Density": abs_dens,
        "UsThem_Pronoun_Density": usthem_dens
    }


def cohens_d(x0: Iterable[float], x1: Iterable[float]) -> float:
    """
    Computes pooled-variance standardized Cohen's d effect size (x1 - x0).
    """
    arr0 = np.asarray(list(x0), dtype=float)
    arr1 = np.asarray(list(x1), dtype=float)
    s0, s1 = np.std(arr0, ddof=1), np.std(arr1, ddof=1)
    n0, n1 = len(arr0), len(arr1)
    pooled = np.sqrt(((n0 - 1) * (s0 ** 2) + (n1 - 1) * (s1 ** 2)) / max(1, (n0 + n1 - 2)))
    return float((np.mean(arr1) - np.mean(arr0)) / (pooled + 1e-12))


if __name__ == "__main__":
    # Quick self-test of the 7-feature Rate-Distortion kernel (W = 50 tokens)
    sample_text = (
        "We must always defend our people completely because they will never stop "
        "attacking our nation and our values. Every single citizen must stand together "
        "against their total hostility, for we are completely united and they are "
        "forever determined to destroy everything we have built in our homeland."
    )
    sample_tokens = clean_and_tokenize(sample_text)
    metrics = extract_rate_distortion_features(sample_tokens, min_content=15)
    print(f"Token count (W): {len(sample_tokens)}")
    for k, v in (metrics or {}).items():
        print(f"  {k:<24}: {v:.4f}")