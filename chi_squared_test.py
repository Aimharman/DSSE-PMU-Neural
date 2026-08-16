"""
===========================================================
chi_squared_test.py

Chi-Squared Bad Data Detection Test

Implements the chi-squared (J2) test for bad data detection
in weighted least squares state estimation. The test statistic
is used to assess the overall consistency of measurements and
identify potential measurement errors.

Reference:
    Abur, A., & Expósito, A. G. (2004). Power System State 
    Estimation: Theory and Implementation. Marcel Dekker.
===========================================================
"""

import numpy as np
from scipy import stats


class ChiSquaredTest:
    """Bad data detection using chi-squared test statistics."""

    def __init__(self, confidence_level=0.95):
        """
        Initialize chi-squared test.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for hypothesis testing (default: 0.95)
            Common values: 0.90, 0.95, 0.99
        """
        self.confidence_level = confidence_level

    def compute_test_statistic(self, residual, R_inv):
        """
        Compute chi-squared test statistic J2.

        The test statistic is:
            J2 = r^T * R_inv * r

        where:
            r = measurement residual vector
            R_inv = inverse of measurement covariance matrix

        Parameters
        ----------
        residual : ndarray
            Residual vector r = z - h(x)

        R_inv : ndarray
            Inverse of measurement covariance matrix

        Returns
        -------
        float
            Chi-squared test statistic J2
        """
        residual = np.asarray(residual, dtype=float)
        R_inv = np.asarray(R_inv, dtype=float)

        J2 = float(residual.T @ R_inv @ residual)

        return max(J2, 0.0)  # Ensure non-negative

    def compute_degrees_of_freedom(self, num_measurements, num_states):
        """
        Compute degrees of freedom for chi-squared distribution.

        df = m - n

        where:
            m = number of measurements
            n = number of state variables

        Parameters
        ----------
        num_measurements : int
            Number of measurements (m)

        num_states : int
            Number of state variables (n)

        Returns
        -------
        int
            Degrees of freedom
        """
        df = max(1, num_measurements - num_states)
        return int(df)

    def compute_critical_value(self, num_measurements, num_states):
        """
        Compute critical value for chi-squared hypothesis test.

        The critical value is the upper quantile of the chi-squared
        distribution at the specified confidence level.

        Parameters
        ----------
        num_measurements : int
            Number of measurements (m)

        num_states : int
            Number of state variables (n)

        Returns
        -------
        float
            Critical value (threshold)
        """
        df = self.compute_degrees_of_freedom(
            num_measurements,
            num_states
        )

        # Upper quantile (one-tailed test)
        alpha = 1.0 - self.confidence_level
        critical = stats.chi2.ppf(1.0 - alpha, df)

        return float(critical)

    def compute_p_value(self, test_statistic, num_measurements, num_states):
        """
        Compute p-value for the test statistic.

        The p-value represents the probability of observing
        a test statistic at least as extreme as the computed
        value, assuming the null hypothesis (no bad data) is true.

        Parameters
        ----------
        test_statistic : float
            Chi-squared test statistic J2

        num_measurements : int
            Number of measurements (m)

        num_states : int
            Number of state variables (n)

        Returns
        -------
        float
            P-value (between 0 and 1)
        """
        df = self.compute_degrees_of_freedom(
            num_measurements,
            num_states
        )

        p_value = float(1.0 - stats.chi2.cdf(test_statistic, df))

        return np.clip(p_value, 0.0, 1.0)

    def test_for_bad_data(self, residual, R_inv, num_measurements, num_states):
        """
        Perform chi-squared test for overall bad data detection.

        Returns a result dictionary containing:
            - test_statistic: Computed J2 value
            - critical_value: Threshold for hypothesis test
            - p_value: Probability of test statistic
            - degrees_of_freedom: Test degrees of freedom
            - bad_data_detected: True if J2 > critical_value
            - confidence: Detection confidence (1 - p_value)

        Parameters
        ----------
        residual : ndarray
            Residual vector r = z - h(x)

        R_inv : ndarray
            Inverse of measurement covariance matrix

        num_measurements : int
            Number of measurements

        num_states : int
            Number of state variables

        Returns
        -------
        dict
            Test results with statistical information
        """
        J2 = self.compute_test_statistic(residual, R_inv)

        df = self.compute_degrees_of_freedom(
            num_measurements,
            num_states
        )

        critical = self.compute_critical_value(
            num_measurements,
            num_states
        )

        p_value = self.compute_p_value(J2, num_measurements, num_states)

        bad_data_detected = J2 > critical

        return {
            "test_statistic": J2,
            "critical_value": critical,
            "p_value": p_value,
            "degrees_of_freedom": df,
            "bad_data_detected": bad_data_detected,
            "anomaly_score": 1.0 - p_value,
            "confidence": 1.0 - p_value,
            "measurements": num_measurements,
            "states": num_states,
        }

    @staticmethod
    def normalized_residuals(residual, R_inv):
        """
        Compute normalized residuals for individual measurement analysis.

        The normalized residual for measurement i is:
            r_norm[i] = residual[i] * sqrt(R_inv[i,i])

        Large normalized residuals indicate measurements that deviate
        significantly from predictions and are candidates for bad data.

        Parameters
        ----------
        residual : ndarray
            Measurement residual vector

        R_inv : ndarray
            Inverse of measurement covariance matrix

        Returns
        -------
        ndarray
            Normalized residuals (magnitude)
        """
        residual = np.asarray(residual, dtype=float)
        R_inv = np.asarray(R_inv, dtype=float)

        r_norm = np.abs(residual) * np.sqrt(np.diag(R_inv))

        return r_norm

    def identify_suspect_measurements(self, residual, R_inv, threshold=3.0):
        """
        Identify individual measurements with suspicious residuals.

        Uses normalized residuals. A measurement is flagged as suspect
        if its normalized residual exceeds the threshold (typically 3.0
        for 3-sigma detection).

        Parameters
        ----------
        residual : ndarray
            Measurement residual vector

        R_inv : ndarray
            Inverse of measurement covariance matrix

        threshold : float, optional
            Detection threshold in standard deviations (default: 3.0)

        Returns
        -------
        dict
            Information about suspect measurements:
            - indices: Array of suspect measurement indices
            - residuals: Normalized residuals for suspects
            - count: Number of suspect measurements
        """
        r_norm = self.normalized_residuals(residual, R_inv)

        suspect_indices = np.flatnonzero(r_norm > threshold)

        return {
            "indices": suspect_indices,
            "residuals": r_norm[suspect_indices] if len(suspect_indices) > 0 else np.array([]),
            "count": len(suspect_indices),
            "normalized_residuals": r_norm,
        }
