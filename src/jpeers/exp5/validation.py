"""Bounded CPU null calibration of the registered noninferiority procedure."""
from __future__ import annotations

import math

import numpy as np

from .analysis import noninferiority_test
from .protocol import derived_seed


def noninferiority_preflight(config: dict) -> dict:
    """Check the implemented boundary test on independent symmetric null contrasts.

    This is a procedure sanity check, not a guarantee that study contrasts obey
    symmetry or a power calculation. Its tolerance and workload are proposals
    which must be recorded/frozen before evaluation.
    """
    statistics = config["statistics"]
    settings = statistics.get("ni_preflight", {})
    n = config["evaluation"]["families"]
    repetitions = settings.get("replications", 500)
    permutations = settings.get("permutations", min(1024, statistics["permutations"]))
    max_products = settings.get("max_sign_products", 200_000_000)
    tolerance = settings.get("tolerance", .02)
    alpha = statistics.get("alpha", .05)
    margin = config["development"]["gates"]["preservation_margin"]
    if (type(n) is not int or n < 2 or type(repetitions) is not int or repetitions < 1
            or type(permutations) is not int or permutations < 1
            or type(max_products) is not int or max_products < 1
            or not 0 < alpha < 1 or not 0 <= tolerance < 1 - alpha or not 0 < margin < 1):
        raise ValueError("Invalid noninferiority preflight parameters")
    assignments = 2 ** n if n <= 20 else permutations
    estimated_products = 2 * repetitions * assignments * n
    report = dict(passed=False, model_calls=0, n_families=n, replications=repetitions,
                  permutations=permutations, exact=n <= 20, margin=margin,
                  alpha=alpha, calibration_tolerance=tolerance,
                  estimated_sign_products=estimated_products, max_sign_products=max_products,
                  distributions=[], seed=int(statistics.get("seed", 20261108)),
                  assumptions=["independent family contrasts", "symmetric contrasts at the null boundary"],
                  scope="procedure null calibration; not target-model validation or a power guarantee")
    if estimated_products > max_products:
        report["reason"] = "Null calibration exceeds its declared CPU work allowance"
        return report
    for kind in ("uniform", "rademacher"):
        rejections = 0
        for iteration in range(repetitions):
            seed = derived_seed(report["seed"], "NI_preflight", kind, iteration) % (2 ** 32)
            rng = np.random.default_rng(seed)
            centered = (rng.uniform(-.2, .2, n) if kind == "uniform"
                        else rng.choice([-.2, .2], n))
            # Under H0 the mean is exactly the -margin boundary in distribution.
            result = noninferiority_test((centered - margin).tolist(), margin=margin,
                                         seed=seed, n_permutations=permutations, n_bootstrap=1)
            rejections += result["p"] <= alpha
        rate = rejections / repetitions
        z = 1.6448536269514722  # one-sided 95% Wilson upper bound
        denominator = 1 + z * z / repetitions
        upper = (rate + z * z / (2 * repetitions)
                 + z * math.sqrt(rate * (1 - rate) / repetitions
                                 + z * z / (4 * repetitions ** 2))) / denominator
        report["distributions"].append(dict(name=kind, replications=repetitions,
                                            rejections=rejections, rejection_rate=rate,
                                            upper_95=float(upper), passed=upper <= alpha + tolerance))
    report["passed"] = repetitions >= 100 and all(r["passed"] for r in report["distributions"])
    if not report["passed"]:
        report["reason"] = "Insufficient precision or excessive simulated boundary rejection"
    return report
