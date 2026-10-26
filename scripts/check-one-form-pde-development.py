#!/usr/bin/env python3
"""Numerically audit the saved one-form PDE experiment, without rerunning it."""
import csv
from pathlib import Path

import numpy as np

BASE = Path("out/c51-one-form-pde")


def metrics(directory):
    with (directory / "metrics.csv").open() as stream:
        return {r["quantity"]: float(r["value"]) for r in csv.DictReader(stream)}


def moments(directory):
    covariance = np.loadtxt(directory / "surface_covariance.csv", delimiter=",")
    mean = np.loadtxt(directory / "surface.csv", delimiter=",", skiprows=1)[:, 1]
    return mean, covariance


def relative_error(candidate, reference):
    return np.linalg.norm(candidate-reference) / np.linalg.norm(reference)


def check_psd(matrix, scale):
    assert np.isfinite(matrix).all()
    assert np.linalg.norm(matrix-matrix.T) <= 1e-9 * scale
    assert np.linalg.eigvalsh((matrix+matrix.T)/2).min() >= -1e-9 * scale


def main():
    exact_errors = []
    checked = 0
    directories = ["selected-scaling", "sparse-selected-repeated", "supported-reference-extension",
                   "spectral-contract-check", "fixed-parameter-scaling"]
    for name in directories:
        folder = BASE / name
        for path in sorted(folder.glob("*/surface_covariance.csv")):
            directory = path.parent
            metadata = metrics(directory)
            mean, covariance = moments(directory)
            assert np.isfinite(mean).all()
            check_psd(covariance, np.linalg.norm(covariance))
            stem, repeat = directory.name.rsplit("-rep", 1)
            method, level = stem.rsplit("-n", 1)
            if repeat != "0":
                original_mean, original_covariance = moments(folder / f"{stem}-rep0")
                assert relative_error(mean, original_mean) < 1e-8, directory
                assert relative_error(covariance, original_covariance) < 1e-8, directory
            reference_dir = folder / f"supported-n{level}-rep0"
            if reference_dir.exists() and (method == "gp" or method.startswith("spectral-")):
                reference_mean, reference_covariance = moments(reference_dir)
                assert metadata["precision_scale"] == metrics(reference_dir)["precision_scale"]
                if method.startswith("spectral-"):
                    # Fixed-tau spectral truncation must underestimate covariance
                    # in Loewner order, including correlated output combinations.
                    check_psd(reference_covariance-covariance, np.linalg.norm(reference_covariance))
                    assert metadata["eigenbasis_relative_residual"] < 1e-7
                    assert metadata["eigenbasis_mass_orthogonality_defect"] < 1e-8
                if method == "gp" or int(method[9:]) == int(metadata["state_dimension"]):
                    errors = relative_error(mean, reference_mean), relative_error(covariance, reference_covariance)
                    assert max(errors) < 1e-8, (directory, errors)
                    exact_errors.append((directory.name, *errors))
            if metadata.get("fixed_precision_scale") == 1 and method in {"compact", "supported", "gp"}:
                original_mean, original_covariance = moments(BASE / "selected-scaling" / f"{method}-n{level}-rep0")
                assert relative_error(mean, original_mean) < 1e-8
                assert relative_error(covariance, original_covariance) < 1e-8
            checked += 1
    assert exact_errors, "No exact-law comparisons found"
    print(f"Audited {checked} completed runs: finite symmetric PSD output covariances; reproducible moments; unchanged fixed-scale laws.")
    print(f"Exact-law checks: {len(exact_errors)}; maximum mean discrepancy {max(r[1] for r in exact_errors):.3e}; maximum covariance discrepancy {max(r[2] for r in exact_errors):.3e}.")
    print("Spectral truncations satisfy the expected covariance ordering within numerical tolerance.")


if __name__ == "__main__":
    main()
