import numpy as np


def localize_dominant_pmu(residual_vector, measurement_covariance=None):
    """Return the dominant PMU from the residual energy distribution.

    For PMU k, J_k = r_k^T W_k r_k and J = sum_k J_k. The PMU with the largest
    residual share is treated as dominant. This is the classical PMU-localization
    heuristic retained in the final system.
    """
    res = np.asarray(residual_vector, dtype=float).reshape(-1)
    if res.size == 0:
        return 0, np.array([]), 0.0

    if measurement_covariance is None:
        weights = np.ones_like(res)
    else:
        weights = np.diag(np.asarray(measurement_covariance, dtype=float))

    pmu_groups = {
        1: res[0:4],
        2: res[4:8],
        3: res[8:12],
    }
    energies = {}
    for pmu_id, group in pmu_groups.items():
        if group.size == 0:
            energies[pmu_id] = 0.0
            continue
        w = weights[0:4] if pmu_id == 1 else weights[4:8] if pmu_id == 2 else weights[8:12]
        energies[pmu_id] = float(group.T @ (w * group))

    total = sum(energies.values()) or 1.0
    shares = {pmu_id: energy / total for pmu_id, energy in energies.items()}
    dominant = max(energies, key=lambda k: energies[k])
    return dominant, np.array([energies[1], energies[2], energies[3]]), shares[dominant]
