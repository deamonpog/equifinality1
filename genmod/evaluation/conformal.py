"""Conformal prediction for inverse generative modeling.

Implements RAPS (Regularized Adaptive Prediction Sets) from
Angelopoulos et al. (2021) "Uncertainty Sets for Image Classifiers
using Conformal Prediction."

Given a trained classifier's softmax outputs, produces prediction sets
with distribution-free coverage guarantees:
    P(true_generator ∈ prediction_set) ≥ 1 - α
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_nonconformity_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    method: str = "raps",
    k_reg: int = 5,
    lambda_reg: float = 0.01,
) -> np.ndarray:
    """Compute non-conformity scores for calibration samples.

    For APS: score = sum of sorted probabilities until the true class is included.
    For RAPS: APS score + regularization penalty for large sets.

    Args:
        probs: (N, C) softmax probabilities.
        labels: (N,) true class labels.
        method: "aps" or "raps".
        k_reg: RAPS regularization starts after k_reg classes.
        lambda_reg: RAPS regularization strength.

    Returns: (N,) non-conformity scores.
    """
    N, C = probs.shape
    scores = np.zeros(N)

    for i in range(N):
        # Sort probabilities in descending order
        sorted_indices = np.argsort(-probs[i])
        sorted_probs = probs[i][sorted_indices]

        # Cumulative sum
        cumsum = np.cumsum(sorted_probs)

        # Find where the true label appears in the sorted order
        true_rank = np.where(sorted_indices == labels[i])[0][0]

        # Randomized APS score (Romano et al. 2020; Angelopoulos et al. 2021):
        #
        #     E = sum_{j ranked above y} p_j  +  u * p_y,    u ~ U(0, 1)
        #
        # Equivalently, the cumulative probability through the true class minus
        # a uniform draw on (0, p_y). The randomization must be applied at
        # EVERY rank, including rank 0.
        #
        # It was previously skipped when the true class was ranked first, which
        # assigned those samples the full p_max instead of u * p_max. For an
        # accurate model most calibration samples ARE rank 0, so their scores
        # were systematically inflated, q_hat was inflated with them, and every
        # prediction set came out too large: coverage sat near 0.97 regardless
        # of alpha, and set size barely responded to alpha at all.
        score = cumsum[true_rank] - np.random.uniform(0, sorted_probs[true_rank])

        # RAPS regularization
        if method == "raps":
            penalty = lambda_reg * max(0, true_rank + 1 - k_reg)
            score += penalty

        scores[i] = score

    return scores


def conformal_calibrate(
    scores: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute the conformal threshold q_hat.

    q_hat = ceil((n+1)(1-α))/n -th quantile of calibration scores.
    Guarantees marginal coverage ≥ 1-α.

    Returns: threshold value.
    """
    n = len(scores)
    quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
    quantile_level = min(quantile_level, 1.0)
    q_hat = np.quantile(scores, quantile_level)
    return q_hat


def conformal_predict(
    probs: np.ndarray,
    q_hat: float,
    method: str = "raps",
    k_reg: int = 5,
    lambda_reg: float = 0.01,
    randomized: bool = True,
    allow_empty: bool = False,
) -> List[np.ndarray]:
    """Construct prediction sets for test samples.

    Classes are added in descending probability order until the score exceeds
    q_hat. The score MUST be built the same way it was during calibration,
    otherwise the coverage guarantee is not the one that was calibrated for.

    randomized:
        Include the class that crosses the threshold only to the extent that a
        uniform draw says it belongs, mirroring the u * p_y term in the
        calibration score (compute_nonconformity_scores). This is the standard
        randomized APS/RAPS construction and yields coverage close to the
        nominal 1 - alpha.

        With randomized=False the crossing class is ALWAYS included, which is
        the conservative construction: coverage still satisfies the >= 1 - alpha
        guarantee, but overshoots it, and the overshoot is large when a single
        class carries much of the mass. On an 11-class problem this produced
        0.91 coverage against a 0.80 target.

    allow_empty:
        The randomized rule can return an empty set when the top class alone
        already exceeds q_hat. Empty sets are legitimate under the guarantee but
        awkward to interpret, so by default the top-1 class is always retained.
        This costs a little coverage overshoot and is the usual convention.

    Returns: list of arrays, each containing class indices in the set.
    """
    N, C = probs.shape
    prediction_sets = []

    for i in range(N):
        sorted_indices = np.argsort(-probs[i])
        sorted_probs = probs[i][sorted_indices]
        cumsum = np.cumsum(sorted_probs)

        pred_set = []
        for j in range(C):
            score = cumsum[j]
            if method == "raps":
                score += lambda_reg * max(0, j + 1 - k_reg)

            if score < q_hat:
                # This class is inside the threshold outright.
                pred_set.append(sorted_indices[j])
                continue

            # This class crosses the threshold. Under the randomized rule it is
            # kept only if the same u * p_j term used in calibration leaves it
            # below q_hat.
            if not randomized:
                pred_set.append(sorted_indices[j])
            else:
                u = np.random.uniform(0, 1)
                if score - u * sorted_probs[j] <= q_hat:
                    pred_set.append(sorted_indices[j])
            break

        if not pred_set and not allow_empty:
            pred_set = [sorted_indices[0]]

        prediction_sets.append(np.array(pred_set))

    return prediction_sets


def evaluate_conformal(
    prediction_sets: List[np.ndarray],
    true_labels: np.ndarray,
    group_labels: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate conformal prediction sets.

    Returns dict with coverage, set sizes, and conditional coverage.
    """
    N = len(prediction_sets)
    set_sizes = np.array([len(s) for s in prediction_sets])

    # Coverage
    covered = np.array([true_labels[i] in prediction_sets[i] for i in range(N)])
    coverage = covered.mean()

    results = {
        "coverage": float(coverage),
        "avg_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "max_set_size": int(set_sizes.max()),
        "min_set_size": int(set_sizes.min()),
        "singleton_fraction": float((set_sizes == 1).mean()),
    }

    # Conditional coverage by group
    if group_labels is not None:
        unique_groups = np.unique(group_labels)
        cond_coverage = {}
        cond_set_size = {}
        for g in unique_groups:
            mask = group_labels == g
            if mask.sum() > 0:
                g_key = str(g)
                cond_coverage[g_key] = float(covered[mask].mean())
                cond_set_size[g_key] = float(set_sizes[mask].mean())
        results["conditional_coverage"] = cond_coverage
        results["conditional_set_size"] = cond_set_size

    return results


class ConformalPredictor:
    """Convenience class wrapping calibration + prediction.

    Usage:
        cp = ConformalPredictor(method="raps", alpha=0.1)
        cp.calibrate(cal_probs, cal_labels)
        sets = cp.predict(test_probs)
        metrics = cp.evaluate(sets, test_labels, group_labels)
    """

    def __init__(
        self,
        method: str = "raps",
        alpha: float = 0.1,
        k_reg: int = 5,
        lambda_reg: float = 0.01,
    ):
        self.method = method
        self.alpha = alpha
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.q_hat: Optional[float] = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Calibrate on held-out data. Returns learned threshold."""
        scores = compute_nonconformity_scores(
            probs, labels, self.method, self.k_reg, self.lambda_reg
        )
        self.q_hat = conformal_calibrate(scores, self.alpha)
        return self.q_hat

    def predict(self, probs: np.ndarray) -> List[np.ndarray]:
        """Produce prediction sets for new data."""
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() first")
        return conformal_predict(
            probs, self.q_hat, self.method, self.k_reg, self.lambda_reg
        )

    def evaluate(
        self,
        prediction_sets: List[np.ndarray],
        true_labels: np.ndarray,
        group_labels: Optional[np.ndarray] = None,
    ) -> dict:
        """Evaluate prediction sets."""
        return evaluate_conformal(prediction_sets, true_labels, group_labels)

    def calibrate_and_evaluate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
        test_probs: np.ndarray,
        test_labels: np.ndarray,
        test_group_labels: Optional[np.ndarray] = None,
    ) -> dict:
        """One-shot: calibrate, predict, evaluate."""
        self.calibrate(cal_probs, cal_labels)
        sets = self.predict(test_probs)
        return self.evaluate(sets, test_labels, test_group_labels)


def run_conformal_at_multiple_alphas(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_group_labels: Optional[np.ndarray] = None,
    alpha_levels: List[float] = [0.01, 0.05, 0.10, 0.20],
    method: str = "raps",
    k_reg: int = 5,
    lambda_reg: float = 0.01,
) -> Dict[float, dict]:
    """Run conformal prediction at multiple alpha levels.

    Returns: dict mapping alpha -> evaluation results.
    """
    results = {}
    for alpha in alpha_levels:
        cp = ConformalPredictor(method=method, alpha=alpha, k_reg=k_reg, lambda_reg=lambda_reg)
        result = cp.calibrate_and_evaluate(
            cal_probs, cal_labels, test_probs, test_labels, test_group_labels
        )
        result["alpha"] = alpha
        result["q_hat"] = cp.q_hat
        results[alpha] = result
    return results
