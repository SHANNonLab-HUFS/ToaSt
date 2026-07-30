"""Head-dimension importance scores for Structured Coupled Weight Pruning.

A head's Q/K pair and its V/O pair are pruned together, so importance is scored on the
*coupled* weight matrices rather than on any single projection. Given head `h` of block `l`,
we form

    W_QK = [ W_Q^h | W_K^h ]     and     W_VO = [ W_V^h | (W_O^h)^T ]

both of shape (head_dim, 2 * embed_dim), and score each of the `head_dim` rows. Pruning row
`i` of W_QK removes one dimension from that head's attention logits; pruning row `i` of W_VO
removes one dimension from its value subspace. Because the score is computed on the
concatenation, a dimension survives only if it matters to *both* halves of the pair.

`gm` (geometric-median distance) is the score used in the paper; `l1` and `l2` are the
magnitude baselines it is compared against. The single-sided and averaged couplings exist
for the ablation that motivates the coupled formulation.
"""

from typing import Callable, Dict, Tuple

import torch

__all__ = [
    "SCORES",
    "COUPLINGS",
    "gm_score",
    "l1_score",
    "l2_score",
    "head_importance",
]


# --------------------------------------------------------------------------------------
# Row scores.  Each takes a (head_dim, D) matrix and returns a (head_dim,) score vector;
# larger means more important.
# --------------------------------------------------------------------------------------


def gm_score(W: torch.Tensor) -> torch.Tensor:
    """Distance of each row from the column-wise median of `W`.

    Rows near the median are close to redundant: the head can reconstruct them from its
    remaining dimensions, so they are the cheapest to drop.
    """
    median = torch.median(W, dim=0)[0]
    return torch.sqrt(torch.sum((W - median) ** 2, dim=1))


def l1_score(W: torch.Tensor) -> torch.Tensor:
    """Row-wise L1 magnitude."""
    return W.abs().sum(dim=1)


def l2_score(W: torch.Tensor) -> torch.Tensor:
    """Row-wise squared L2 magnitude."""
    return W.pow(2.0).sum(dim=1)


SCORES: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gm": gm_score,
    "l1": l1_score,
    "l2": l2_score,
}


# --------------------------------------------------------------------------------------
# Couplings.  Each takes one head's four projections and returns (qk_importance,
# vo_importance), both (head_dim,).  `O_T` is the transpose of the output-projection
# columns belonging to this head, so all four arrive as (head_dim, embed_dim).
# --------------------------------------------------------------------------------------


def _coupled(score, Q, K, V, O_T):
    """Paper default: score the concatenated pairs [Q|K] and [V|O^T]."""
    return (
        score(torch.cat([Q, K], dim=1)),
        score(torch.cat([V, O_T], dim=1)),
    )


def _q_only(score, Q, K, V, O_T):
    """Ablation: score only the upstream half of each pair."""
    return score(Q), score(V)


def _k_only(score, Q, K, V, O_T):
    """Ablation: score only the downstream half of each pair."""
    return score(K), score(O_T)


def _proj_only(score, Q, K, V, O_T):
    """Ablation: Q for the attention pair, output projection for the value pair."""
    return score(Q), score(O_T)


def _average(score, Q, K, V, O_T):
    """Ablation: score each projection separately, then average within the pair."""
    return (
        (score(Q) + score(K)) / 2.0,
        (score(V) + score(O_T)) / 2.0,
    )


COUPLINGS: Dict[str, Callable] = {
    "coupled": _coupled,
    "q_only": _q_only,
    "k_only": _k_only,
    "proj_only": _proj_only,
    "average": _average,
}


def head_importance(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O_T: torch.Tensor,
    score: str = "gm",
    coupling: str = "coupled",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Importance of each dimension of a single attention head.

    Args:
        Q, K, V: this head's rows of the Q, K and V projections, each (head_dim, embed_dim).
        O_T: this head's columns of the output projection, transposed to (head_dim, embed_dim).
        score: one of ``SCORES`` -- ``gm`` (default), ``l1`` or ``l2``.
        coupling: one of ``COUPLINGS`` -- ``coupled`` (default) or an ablation variant.

    Returns:
        ``(qk_importance, vo_importance)``, each of shape ``(head_dim,)``.
    """
    if score not in SCORES:
        raise ValueError(f"unknown score {score!r}; choose from {sorted(SCORES)}")
    if coupling not in COUPLINGS:
        raise ValueError(f"unknown coupling {coupling!r}; choose from {sorted(COUPLINGS)}")
    return COUPLINGS[coupling](SCORES[score], Q, K, V, O_T)
