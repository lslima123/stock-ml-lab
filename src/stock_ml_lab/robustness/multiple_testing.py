from __future__ import annotations

from math import comb

import numpy as np


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted q-values.

    NaN inputs remain NaN and do not enter the multiplicity count.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional.")

    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    values = p[finite]

    if values.size == 0:
        return out
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Finite p-values must lie in [0, 1].")

    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)

    adjusted_ranked = ranked * m / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    restored = np.empty_like(adjusted_ranked)
    restored[order] = adjusted_ranked
    out[finite] = restored
    return out


def exact_sign_test_greater(wins: int, total: int, p0: float = 0.5) -> float:
    """Exact one-sided binomial/sign-test p-value P[X >= wins]."""
    if total < 1:
        raise ValueError("total must be >= 1.")
    if not 0 <= wins <= total:
        raise ValueError("wins must satisfy 0 <= wins <= total.")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must lie in [0, 1].")

    return float(
        sum(
            comb(total, k) * (p0**k) * ((1.0 - p0) ** (total - k))
            for k in range(wins, total + 1)
        )
    )
