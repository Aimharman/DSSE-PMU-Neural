"""
===========================================================
decision_fusion.py

Decision Fusion Module

Fuses outputs from:
  - Neural Stream: Fault detection + confidence
  - Classical Stream: WLS state estimation + Chi-squared test

Implements agreement/disagreement logic to make final decisions
on data quality and measurement management actions.

Topology:
                    PMU DATA
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Neural Stream                Classical Stream
         │                             │
   Fault + Confidence              WLS +
                                   χ² Test
         │                             │
         └──────────────┬──────────────┘
                        ▼
                  Decision Fusion
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    AGREEMENT      DISAGREEMENT      NORMAL
         │              │              │
         ▼              ▼              ▼
   HIGH CONFIDENCE  INVESTIGATE    ACCEPT
     ACTION         / VALIDATE
===========================================================
"""

import numpy as np
from enum import Enum


class FusionDecision(Enum):
    """Possible fusion decision outcomes."""

    AGREEMENT = "AGREEMENT"  # Both systems agree on fault
    DISAGREEMENT = "DISAGREEMENT"  # Systems disagree
    NORMAL = "NORMAL"  # Both systems indicate normal operation


class DecisionFusion:
    """Fuses neural and classical stream decisions."""

    def __init__(
        self,
        neural_confidence_threshold=0.70,
        chi2_significance_level=0.05,
    ):
        """
        Initialize decision fusion engine.

        Parameters
        ----------
        neural_confidence_threshold : float
            Minimum confidence for neural prediction (default: 0.70)

        chi2_significance_level : float
            Significance level for chi-squared test (default: 0.05)
            Equivalently, confidence level = 1 - significance_level = 0.95
        """
        self.neural_threshold = neural_confidence_threshold
        self.chi2_alpha = chi2_significance_level

    def _interpret_neural_result(self, neural_result):
        """
        Interpret neural controller result.

        Returns
        -------
        dict
            - fault_detected: bool, True if fault found
            - fault_type: str, type of fault detected
            - confidence: float, confidence in prediction
            - valid: bool, whether result is valid/reliable
        """
        # Neural streams may have different result structures
        # Try to extract key information
        fault_type = neural_result.get("fault_type", "UNKNOWN")
        confidence = neural_result.get("confidence", 0.0)

        # A fault is detected if:
        # 1. Result is not NORMAL
        # 2. Confidence exceeds threshold
        fault_detected = (
            fault_type != "NORMAL" and
            confidence >= self.neural_threshold
        )

        valid = confidence >= self.neural_threshold

        return {
            "fault_detected": fault_detected,
            "fault_type": fault_type,
            "confidence": confidence,
            "valid": valid,
        }

    def _interpret_chi2_result(self, chi2_result):
        """
        Interpret chi-squared test result.

        Note:
            The classical chi-square test detects overall statistical
            inconsistency, not fault type or PMU localization. It should be
            treated as a validation signal, not a neural-classifier substitute.
        """
        test_stat = chi2_result.get("test_statistic", 0.0)
        critical = chi2_result.get("critical_value", 0.0)
        p_value = chi2_result.get("p_value", 1.0)
        detected = chi2_result.get("bad_data_detected", False)
        anomaly_score = chi2_result.get("anomaly_score", 1.0 - p_value)
        status = "CONSISTENT" if not detected else "INCONSISTENT"

        return {
            "bad_data_detected": detected,
            "test_statistic": test_stat,
            "critical_value": critical,
            "p_value": p_value,
            "anomaly_score": anomaly_score,
            "statistical_status": status,
            "classical_decision": "ANOMALY" if detected else "NORMAL",
            "confidence": anomaly_score,
        }

    def fuse(self, neural_result, chi2_result):
        """
        Perform decision fusion on neural and classical results.

        Parameters
        ----------
        neural_result : dict
            Output from neural controller containing:
            - fault_type: str
            - confidence: float
            - (other fields preserved)

        chi2_result : dict
            Output from chi-squared test containing:
            - bad_data_detected: bool
            - test_statistic: float
            - critical_value: float
            - p_value: float
            - degrees_of_freedom: int
            - (other fields preserved)

        Returns
        -------
        dict
            Fusion decision with:
            - decision: FusionDecision enum
            - decision_name: str
            - agreement: bool
            - neural_interpretation: dict
            - chi2_interpretation: dict
            - confidence: float
            - action: str
            - reasoning: str
        """
        # Interpret results from both streams
        neural_interp = self._interpret_neural_result(neural_result)
        chi2_interp = self._interpret_chi2_result(chi2_result)

        # Determine agreement
        neural_fault = neural_interp["fault_detected"]
        chi2_fault = chi2_interp["bad_data_detected"]

        agreement = (neural_fault == chi2_fault)

        # Compute decision
        if agreement:
            if neural_fault and chi2_fault:
                decision = FusionDecision.AGREEMENT
                confidence = min(
                    neural_interp["confidence"],
                    chi2_interp["anomaly_score"],
                )
                action = "HIGH CONFIDENCE ACTION"
                reasoning = (
                    "Neural fault diagnosis and the classical chi-square test "
                    "both indicate a statistically inconsistent measurement set. "
                    "Retain the neural PMU localization and apply mitigation."
                )
            else:
                decision = FusionDecision.NORMAL
                confidence = min(
                    neural_interp["confidence"],
                    1.0 - chi2_interp["p_value"],
                )
                action = "ACCEPT ALL DATA"
                reasoning = (
                    "Both streams indicate a consistent normal operating state. "
                    "Continue without PMU mitigation."
                )
        else:
            decision = FusionDecision.DISAGREEMENT
            confidence = min(
                neural_interp["confidence"],
                chi2_interp["anomaly_score"],
            )
            action = "INVESTIGATE / VALIDATE"
            reasoning = (
                "The neural classifier reports a structured anomaly, but the "
                "aggregate classical chi-square test does not exceed its threshold. "
                "This is a genuine disagreement and should be reviewed."
            )

        return {
            "decision": decision,
            "decision_name": decision.value,
            "agreement": agreement,
            "neural_interpretation": neural_interp,
            "chi2_interpretation": chi2_interp,
            "confidence": confidence,
            "action": action,
            "reasoning": reasoning,
            "neural_fault_type": neural_interp["fault_type"],
            "neural_confidence": neural_interp["confidence"],
            "chi2_test_statistic": chi2_interp["test_statistic"],
            "chi2_critical_value": chi2_interp["critical_value"],
            "chi2_p_value": chi2_interp["p_value"],
        }

    def print_fusion_report(self, fusion_result):
        """
        Print a formatted fusion decision report.

        Parameters
        ----------
        fusion_result : dict
            Output from fuse() method
        """
        print("\n" + "=" * 80)
        print(" DECISION FUSION REPORT")
        print("=" * 80)

        print(f"\nDecision          : {fusion_result['decision_name']}")
        print(f"Agreement         : {'YES' if fusion_result['agreement'] else 'NO'}")
        print(f"Overall Confidence: {fusion_result['confidence']:.4f}")

        print("\n" + "-" * 80)
        print(" NEURAL STREAM")
        print("-" * 80)
        neural = fusion_result["neural_interpretation"]
        print(f"Fault Detected    : {neural['fault_detected']}")
        print(f"Fault Type        : {neural['fault_type']}")
        print(f"Confidence        : {neural['confidence']:.4f}")
        print(f"Valid             : {neural['valid']}")

        print("\n" + "-" * 80)
        print(" CLASSICAL STREAM (CHI-SQUARED TEST)")
        print("-" * 80)
        chi2 = fusion_result["chi2_interpretation"]
        print(f"Bad Data Detected : {chi2['bad_data_detected']}")
        print(f"Test Statistic    : {chi2['test_statistic']:.8f}")
        print(f"Critical Value    : {chi2['critical_value']:.8f}")
        print(f"P-value           : {chi2['p_value']:.6f}")
        print(f"Statistical Status: {chi2['statistical_status']}")
        print(f"Anomaly Score     : {chi2['anomaly_score']:.4f}")

        print("\n" + "-" * 80)
        print(" FUSION DECISION")
        print("-" * 80)
        print(f"Action            : {fusion_result['action']}")
        print(f"Reasoning         : {fusion_result['reasoning']}")

        print("\n" + "=" * 80)
