from src.feature_extraction import compute_window_metrics

tokens = ["sample", "tokenized", "words", "..."]
features = compute_window_metrics(tokens)

# Returns:
# features['D_KL_Global']          : Kullback-Leibler distribution divergence
# features['D_KL_Tail_Deconfounded']: Topic-deconfounded tail divergence
# features['Yules_K']              : Length-invariant lexical repeat rate
# features['R_LZ_Content']         : Lempel-Ziv algorithmic redundancy
# features['Bigram_Redundancy']    : Sequential phrase repetition