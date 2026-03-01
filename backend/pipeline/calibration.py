import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

CACHE_DIR              = Path("/tmp/drug_repurposing_cache")
CALIBRATION_PARAMS_FILE = CACHE_DIR / "calibration_params.json"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_CALIBRATION_BINS = 10  # for ECE computation
MIN_SAMPLES_FOR_CALIBRATION = 20


class ScoreCalibrator:
    """
    Calibrates drug repurposing composite scores.

    Two calibration methods:
      - Platt scaling (logistic regression on scores → probabilities)
      - Isotonic regression (non-parametric, better for bimodal distributions)

    Usage
    -----
        calibrator = ScoreCalibrator()
        calibrator.fit(train_scores, train_labels)   # label=1 if TP, 0 if TN
        prob = calibrator.transform(raw_score)
        summary = calibrator.calibration_summary(val_scores, val_labels)
    """

    def __init__(self, method: str = "platt"):
        """
        Parameters
        ----------
        method : str
            "platt"    — logistic regression (sigmoid fit)
            "isotonic" — isotonic regression (non-parametric)
        """
        if method not in ("platt", "isotonic"):
            raise ValueError(f"method must be 'platt' or 'isotonic', got '{method}'")

        self.method        = method
        self._fitted       = False
        self._platt_a      = None   # Platt scaling parameter A
        self._platt_b      = None   # Platt scaling parameter B
        self._isotonic     = None   # IsotonicRegression instance
        self._fit_hash     = None   # hash of training data
        self._fit_time     = None   # when calibration was run

        # Auto-load saved params if available
        self._try_load_params()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _try_load_params(self) -> None:
        """Attempt to load calibration parameters from disk."""
        if not CALIBRATION_PARAMS_FILE.exists():
            logger.debug("No saved calibration params found — run fit() first.")
            return

        try:
            params = json.loads(CALIBRATION_PARAMS_FILE.read_text())
            method = params.get("method", "platt")

            if method != self.method:
                logger.info(
                    f"Saved calibration method '{method}' ≠ requested '{self.method}' "
                    f"— ignoring saved params."
                )
                return

            if method == "platt":
                self._platt_a  = params["platt_a"]
                self._platt_b  = params["platt_b"]
            elif method == "isotonic":
                # IsotonicRegression can't be saved to JSON directly;
                # store threshold breakpoints
                self._iso_X = np.array(params["iso_X"])
                self._iso_y = np.array(params["iso_y"])
                iso         = IsotonicRegression(out_of_bounds="clip")
                iso.fit(self._iso_X, self._iso_y)
                self._isotonic = iso

            self._fitted   = True
            self._fit_hash = params.get("fit_hash")
            self._fit_time = params.get("fit_time")
            logger.info(
                f"✅ Loaded calibration params ({method}), "
                f"fitted {params.get('fit_time', 'unknown')} "
                f"on {params.get('n_samples', '?')} samples"
            )
        except Exception as e:
            logger.warning(f"Failed to load calibration params: {e}")

    def save_params(self) -> None:
        """Save calibration parameters to disk."""
        if not self._fitted:
            logger.warning("Cannot save — calibrator not fitted yet.")
            return

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        params: Dict = {
            "method":    self.method,
            "fit_hash":  self._fit_hash,
            "fit_time":  self._fit_time,
        }

        if self.method == "platt":
            params["platt_a"] = self._platt_a
            params["platt_b"] = self._platt_b
        elif self.method == "isotonic" and self._isotonic is not None:
            # Save breakpoints for serialisation
            params["iso_X"] = list(self._isotonic.X_thresholds_)
            params["iso_y"] = list(self._isotonic.y_thresholds_)

        try:
            CALIBRATION_PARAMS_FILE.write_text(json.dumps(params, indent=2))
            logger.info(f"Calibration params saved → {CALIBRATION_PARAMS_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save calibration params: {e}")

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        scores: List[float],
        labels: List[int],
        n_samples: Optional[int] = None,
    ) -> "ScoreCalibrator":
        """
        Fit calibration model.

        Parameters
        ----------
        scores : list of float
            Raw composite scores (0–1).
        labels : list of int
            1 = known positive (TP drug–disease pair), 0 = known negative.
        n_samples : int, optional
            Total number of validation samples (for logging).
        """
        scores_np = np.array(scores).reshape(-1, 1)
        labels_np = np.array(labels)

        if len(scores_np) < MIN_SAMPLES_FOR_CALIBRATION:
            raise ValueError(
                f"Need at least {MIN_SAMPLES_FOR_CALIBRATION} samples for calibration, "
                f"got {len(scores_np)}."
            )

        # Hash the training data for versioning
        data_str    = json.dumps({"s": scores[:50], "l": labels[:50]})
        self._fit_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        self._fit_time = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

        if self.method == "platt":
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(scores_np, labels_np)
            self._platt_a = float(lr.coef_[0][0])
            self._platt_b = float(lr.intercept_[0])
            logger.info(
                f"✅ Platt calibration fitted: A={self._platt_a:.4f}, "
                f"B={self._platt_b:.4f} on {len(scores)} samples"
            )
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores_np.flatten(), labels_np.astype(float))
            self._isotonic = iso
            logger.info(
                f"✅ Isotonic calibration fitted on {len(scores)} samples, "
                f"{len(iso.X_thresholds_)} breakpoints"
            )

        self._fitted = True
        self.save_params()
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, score: float) -> float:
        """
        Transform a raw composite score to a calibrated probability.
        Returns the raw score if calibrator is not fitted (with a warning).
        """
        if not self._fitted:
            logger.warning(
                "Calibrator not fitted — returning raw score. "
                "Run fit() or load calibration_params.json first."
            )
            return float(score)

        if self.method == "platt":
            logit = self._platt_a * score + self._platt_b
            return float(1.0 / (1.0 + math.exp(-logit)))
        elif self.method == "isotonic" and self._isotonic is not None:
            return float(self._isotonic.predict([score])[0])

        return float(score)

    def transform_batch(self, scores: List[float]) -> List[float]:
        """Transform a list of raw scores to calibrated probabilities."""
        return [self.transform(s) for s in scores]

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_ece(
        self,
        probs:  List[float],
        labels: List[int],
        n_bins: int = N_CALIBRATION_BINS,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|

        Parameters
        ----------
        probs  : calibrated probabilities
        labels : binary labels (1=TP, 0=TN)
        n_bins : number of equal-width bins

        Returns
        -------
        ece : float (lower is better; < 0.10 is well-calibrated)
        """
        probs_np  = np.array(probs)
        labels_np = np.array(labels)
        n         = len(probs_np)
        if n == 0:
            return 0.0

        bins     = np.linspace(0.0, 1.0, n_bins + 1)
        ece      = 0.0
        for i in range(n_bins):
            mask = (probs_np > bins[i]) & (probs_np <= bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc  = labels_np[mask].mean()
            bin_conf = probs_np[mask].mean()
            ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)

        return round(float(ece), 6)

    def compute_brier_score(
        self, probs: List[float], labels: List[int]
    ) -> float:
        """Brier score (MSE of probability predictions; 0 is perfect)."""
        return round(float(brier_score_loss(labels, probs)), 6)

    def compute_auroc(
        self, scores: List[float], labels: List[int]
    ) -> float:
        """Area under the ROC curve."""
        if len(set(labels)) < 2:
            return float("nan")
        return round(float(roc_auc_score(labels, scores)), 6)

    def compute_auprc(
        self, scores: List[float], labels: List[int]
    ) -> float:
        """Area under the Precision-Recall curve."""
        if len(set(labels)) < 2:
            return float("nan")
        return round(float(average_precision_score(labels, scores)), 6)

    def find_optimal_threshold(
        self,
        probs:   List[float],
        labels:  List[int],
        method:  str = "youden",
    ) -> Tuple[float, Dict]:
        """
        Find the decision threshold that optimises sensitivity + specificity.

        Parameters
        ----------
        probs  : calibrated probabilities
        labels : binary labels
        method : "youden" (J = sens + spec - 1) or "f1"

        Returns
        -------
        (optimal_threshold, metrics_at_threshold)
        """
        probs_np  = np.array(probs)
        labels_np = np.array(labels)
        thresholds = np.linspace(0.05, 0.95, 91)

        best_score     = -1.0
        best_threshold = 0.5
        best_metrics   = {}

        for thresh in thresholds:
            preds = (probs_np >= thresh).astype(int)
            tp    = int(((preds == 1) & (labels_np == 1)).sum())
            tn    = int(((preds == 0) & (labels_np == 0)).sum())
            fp    = int(((preds == 1) & (labels_np == 0)).sum())
            fn    = int(((preds == 0) & (labels_np == 1)).sum())

            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            f1   = 2 * prec * sens / max(prec + sens, 1e-8)

            score = (sens + spec - 1.0) if method == "youden" else f1

            if score > best_score:
                best_score     = score
                best_threshold = float(thresh)
                best_metrics   = {
                    "threshold":   round(float(thresh), 3),
                    "sensitivity": round(sens, 4),
                    "specificity": round(spec, 4),
                    "precision":   round(prec, 4),
                    "f1":          round(f1, 4),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                }

        logger.info(
            f"Optimal threshold ({method}): {best_threshold:.2f} → "
            f"sens={best_metrics.get('sensitivity',0):.2f}, "
            f"spec={best_metrics.get('specificity',0):.2f}"
        )
        return best_threshold, best_metrics

    def generate_reliability_diagram_data(
        self,
        probs:  List[float],
        labels: List[int],
        n_bins: int = 10,
    ) -> List[Dict]:
        """
        Generate data for a reliability (calibration) diagram.
        Returns list of {bin_lower, bin_upper, mean_predicted, fraction_positive, n}

        Use this for Figure X in your methods paper.
        """
        probs_np  = np.array(probs)
        labels_np = np.array(labels)
        bins      = np.linspace(0.0, 1.0, n_bins + 1)
        result    = []

        for i in range(n_bins):
            mask = (probs_np > bins[i]) & (probs_np <= bins[i + 1])
            n    = int(mask.sum())
            if n == 0:
                result.append({
                    "bin_lower": round(bins[i], 2),
                    "bin_upper": round(bins[i + 1], 2),
                    "mean_predicted":     None,
                    "fraction_positive":  None,
                    "n": 0,
                })
                continue
            result.append({
                "bin_lower":          round(float(bins[i]), 2),
                "bin_upper":          round(float(bins[i + 1]), 2),
                "mean_predicted":     round(float(probs_np[mask].mean()), 4),
                "fraction_positive":  round(float(labels_np[mask].mean()), 4),
                "n": n,
            })

        return result

    # ── Full summary ──────────────────────────────────────────────────────────

    def calibration_summary(
        self,
        raw_scores:   List[float],
        labels:       List[int],
        name:         str = "validation_set",
    ) -> Dict:
        """
        Compute a full calibration summary for a validation set.

        This is the primary output used in the methods paper metrics table.
        All values are computed dynamically from the provided scores and labels —
        no hardcoded constants.

        Parameters
        ----------
        raw_scores : list of float
            Raw composite scores from the pipeline.
        labels : list of int
            1=known positive (TP), 0=known negative (TN).
        name : str
            Label for this validation set (for logging).

        Returns
        -------
        summary : dict with all calibration metrics
        """
        if not self._fitted:
            logger.warning(
                f"⚠️  calibration_summary called on unfitted calibrator. "
                f"Metrics will be computed on raw scores (uncalibrated). "
                f"Run fit() first for calibrated probabilities."
            )
            cal_probs = raw_scores
            calibrated = False
        else:
            cal_probs  = self.transform_batch(raw_scores)
            calibrated = True

        n_pos  = sum(labels)
        n_neg  = len(labels) - n_pos
        n_tot  = len(labels)

        auroc  = self.compute_auroc(raw_scores, labels)
        auprc  = self.compute_auprc(raw_scores, labels)
        brier  = self.compute_brier_score(cal_probs, labels)
        ece    = self.compute_ece(cal_probs, labels)

        optimal_thresh, thresh_metrics = self.find_optimal_threshold(
            cal_probs, labels, method="youden"
        )

        # Pass/fail thresholds (from validation spec)
        pass_fail = {
            "auroc_pass":      auroc >= 0.70 if not math.isnan(auroc) else None,
            "sensitivity_pass":thresh_metrics.get("sensitivity", 0) >= 0.65,
            "specificity_pass":thresh_metrics.get("specificity", 0) >= 0.75,
            "ece_pass":        ece < 0.15,
            "brier_pass":      brier < 0.25,
        }

        if self._fitted and self.method == "platt":
            calibration_params = {
                "method":    "platt",
                "platt_a":   self._platt_a,
                "platt_b":   self._platt_b,
                "note": (
                    "A < 0 indicates well-calibrated scores (scores increase "
                    "monotonically with true positive rate)."
                ),
            }
        elif self._fitted and self.method == "isotonic":
            calibration_params = {
                "method":         "isotonic",
                "n_breakpoints":  (
                    len(self._isotonic.X_thresholds_)
                    if self._isotonic else None
                ),
            }
        else:
            calibration_params = {
                "method":    "none",
                "note":      "Calibrator not fitted — using raw scores.",
            }

        summary = {
            "name":              name,
            "calibrated":        calibrated,
            "calibration_method": self.method if calibrated else "none",
            "run_timestamp":     time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "fit_timestamp":     self._fit_time,
            "fit_hash":          self._fit_hash,

            "n_total":           n_tot,
            "n_positive":        n_pos,
            "n_negative":        n_neg,
            "prevalence":        round(n_pos / max(n_tot, 1), 4),

            "metrics": {
                "auroc":         auroc,
                "auprc":         auprc,
                "brier_score":   brier,
                "ece":           ece,  # <-- computed dynamically, never hardcoded
                "n_ece_bins":    N_CALIBRATION_BINS,
            },

            "optimal_threshold": optimal_thresh,
            "metrics_at_threshold": thresh_metrics,
            "pass_fail":         pass_fail,
            "all_passed":        all(v for v in pass_fail.values() if v is not None),

            "calibration_params": calibration_params,

            "reliability_diagram": self.generate_reliability_diagram_data(
                cal_probs, labels
            ),
        }

        # Log pass/fail
        status = "✅ PASS" if summary["all_passed"] else "❌ FAIL"
        logger.info(
            f"{status} Calibration summary [{name}]: "
            f"AUROC={auroc:.3f}, "
            f"sens={thresh_metrics.get('sensitivity',0):.3f}, "
            f"spec={thresh_metrics.get('specificity',0):.3f}, "
            f"ECE={ece:.4f} (computed from {n_tot} samples)"
        )

        return summary


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def load_calibrator(method: str = "platt") -> ScoreCalibrator:
    """
    Load a calibrator, auto-loading saved params if available.
    Use this as the default entry point in the pipeline.
    """
    return ScoreCalibrator(method=method)


def calibrate_scores(
    raw_scores: List[float],
    labels:     List[int],
    method:     str = "platt",
) -> Tuple[List[float], Dict]:
    """
    One-shot: fit calibrator and transform scores.

    Returns
    -------
    (calibrated_probs, calibration_summary)
    """
    cal = ScoreCalibrator(method=method)
    cal.fit(raw_scores, labels)
    cal_probs = cal.transform_batch(raw_scores)
    summary   = cal.calibration_summary(raw_scores, labels)
    return cal_probs, summary