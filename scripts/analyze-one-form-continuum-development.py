#!/usr/bin/env python3
"""Audit the independent-forcing, fixed-sensor one-form PDE experiment."""
import argparse
import csv
import math
from pathlib import Path
import statistics

import numpy as np

BASE = Path("out/c51-one-form-pde")
SCALE = 206.6615202953504
ERRORS = (
    "mean_error", "heldout_mean_error", "covariance_error",
    "heldout_covariance_error", "maximum_variance_error",
    "replacement_mean_error", "heldout_replacement_mean_error",
)
LABELS = {"compact": "Compact GMRF", "supported": "Exact supported GMRF",
          "gp": "Exact covariance GP"}
COLORS = {"compact": "#087f73", "supported": "#2864b4", "gp": "#8b50a8"}


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def moments(directory):
    surface = records(directory / "surface.csv")
    mean = np.array([float(r["mean"]) for r in surface])
    truth = np.array([float(r["truth"]) for r in surface])
    cov = np.loadtxt(directory / "surface_covariance.csv", delimiter=",")
    replacements = np.loadtxt(directory / "replacement_surface_means.csv", delimiter=",")
    assert mean.shape == truth.shape == (12,)
    assert cov.shape == (12, 12) and replacements.shape == (20, 12)
    assert all(np.isfinite(x).all() for x in (mean, truth, cov, replacements))
    assert relative(cov, cov.T) < 1e-9, directory
    eigenvalues = np.linalg.eigvalsh((cov + cov.T) / 2)
    assert eigenvalues[0] >= -1e-9 * eigenvalues[-1], directory
    return mean, truth, cov, replacements


def errors(candidate, reference):
    m, truth, c, updates = candidate
    mr, tr, cr, ur = reference
    assert np.array_equal(truth, tr)
    assert np.all(cr.diagonal() > 0)
    values = [relative(m, mr), relative(m[6:], mr[6:]), relative(c, cr),
              relative(c[6:, 6:], cr[6:, 6:]),
              float(np.max(np.abs(c.diagonal() / cr.diagonal() - 1))),
              max(relative(a, b) for a, b in zip(updates, ur)),
              max(relative(a[6:], b[6:]) for a, b in zip(updates, ur))]
    return dict(zip(ERRORS, values)) | {"gate_error": max(values),
                                      "mean_to_analytic_truth_error": relative(m, truth)}


def audit_uq(folder):
    rows = records(folder / "timings.csv")
    assert rows and all(r["status"] == "ok" for r in rows)
    assert len({(r["method"], r["refinement"], r["repeat"]) for r in rows}) == len(rows)
    result, audit = [], []
    truth = None
    for level in sorted({int(r["refinement"]) for r in rows}):
        reference = moments(folder / f"supported-n{level}-rep0")
        if truth is not None:
            assert np.array_equal(truth, reference[1]), "physical truth changed with mesh"
        truth = reference[1]
        methods = sorted({r["method"] for r in rows if int(r["refinement"]) == level})
        for method in methods:
            group = [r for r in rows if int(r["refinement"]) == level and r["method"] == method]
            measured = [r for r in group if r["warmup"] == "False"]
            timing_rows = measured or group
            all_errors = []
            for row in group:
                directory = folder / f"{method}-n{level}-rep{row['repeat']}"
                assert row["workload"] == "continuum"
                assert float(row["continuum_forcing"]) == float(row["fixed_precision_scale"]) == 1
                assert float(row["precision_scale"]) == SCALE
                assert float(row["sensor_rows"]) == 6 and float(row["pde_sigma"]) == .1
                assert float(row["trace_probes"]) == 0
                assert float(row["surface_interpolation_relative_error"]) < 1e-10
                assert float(row["cached_update_maximum_relative_error"]) < 1e-8
                candidate = moments(directory)
                error = errors(candidate, reference)
                n = int(float(row["state_dimension"]))
                if method in {"supported", "gp", f"spectral-{n}"}:
                    assert error["gate_error"] < 1e-8, (directory, error)
                if method.startswith("spectral-"):
                    assert float(row["eigenbasis_relative_residual"]) < 1e-7
                    assert float(row["eigenbasis_mass_orthogonality_defect"]) < 1e-8
                    # A truncated prior must not produce a larger posterior
                    # covariance than the full prior, at identical parameters.
                    diff = reference[2] - candidate[2]
                    assert np.linalg.eigvalsh((diff + diff.T) / 2)[0] >= -1e-8 * np.linalg.norm(reference[2], 2), directory
                all_errors.append(error)
                audit.append(dict(directory=str(directory), **error))
            times = [float(r["selected_uq_total_s"]) for r in timing_rows]
            result.append(dict(
                refinement=level, N=n, method=method,
                repetitions=len(measured), timing_kind="repeated" if measured else "single pilot",
                median_s=statistics.median(times), min_s=min(times), max_s=max(times),
                median_peak_GiB=statistics.median(float(r["process_peak_rss_bytes"]) / 2**30 for r in timing_rows),
                factor_reused_20_means_s=statistics.median(float(r["replacement_means_s"]) for r in timing_rows),
                cached_20_means_s=statistics.median(float(r["cached_replacement_means_s"]) for r in timing_rows),
                **{key: max(e[key] for e in all_errors) for key in all_errors[0]},
            ))
    return result, audit


def audit_pde():
    rows = records(BASE / "continuum-pde-validation/timings.csv")
    result = []
    previous = None
    for row in sorted(rows, key=lambda r: int(r["refinement"])):
        assert row["status"] == "ok" and row["method"] == "pde"
        assert float(row["continuum_forcing"]) == 1
        assert float(row["surface_interpolation_relative_error"]) < 1e-10
        level, error = int(row["refinement"]), float(row["surface_relative_error"])
        assert error > 1e-5, "A zero error would suggest a discrete manufactured load"
        order = np.log(previous[1] / error) / np.log(level / previous[0]) if previous else ""
        if previous:
            assert 1.8 < order < 2.2, (level, order)
        result.append(dict(refinement=level, N=int(float(row["state_dimension"])),
                           flux_relative_error=error, observed_order=order,
                           state_interpolant_relative_l2_error=float(row["state_interpolant_relative_l2_error"]),
                           solve_and_setup_s=float(row["total_compute_s"])))
        previous = level, error
    return result


def tolerance_table(summary):
    result = []
    for n in sorted({r["N"] for r in summary}):
        measured = [r for r in summary if r["N"] == n and r["repetitions"] >= 3]
        for tolerance in (.01, .05):
            for family in ("compact", "supported", "gp", "spectral"):
                eligible = [r for r in measured if r["method"].split("-")[0] == family and r["gate_error"] <= tolerance]
                if eligible:
                    best = min(eligible, key=lambda r: r["median_s"])
                    result.append(dict(N=n, tolerance=tolerance, family=family,
                                       method=best["method"], median_s=best["median_s"],
                                       gate_error=best["gate_error"], median_peak_GiB=best["median_peak_GiB"]))
    return result


def audit_normalization():
    """Change only compact tau in the earlier cochain-sensor workload."""
    result = []
    for level in (6, 8):
        common = BASE / f"compact-normalization-ablation/compact-n{level}-rep0"
        calibrated = BASE / f"fixed-parameter-scaling/compact-n{level}-rep0"
        exact = BASE / f"fixed-parameter-scaling/supported-n{level}-rep0"
        if not common.exists():
            continue
        parameters = lambda d: {r["quantity"]: float(r["value"]) for r in records(d / "metrics.csv")}
        pm, cm, em = [parameters(d) for d in (common, calibrated, exact)]
        assert math.isclose(pm["precision_scale"], em["precision_scale"], rel_tol=1e-14)
        assert pm["continuum_forcing"] == 0
        for key in ("state_dimension", "pde_sigma", "sensor_rows"):
            assert pm[key] == cm[key] == em[key]
        reference_truth = [r["truth"] for r in records(exact / "surface.csv")]
        for d in (common, calibrated):
            assert [r["truth"] for r in records(d / "surface.csv")] == reference_truth
        a, c, r = [np.loadtxt(d / "surface_covariance.csv", delimiter=",") for d in (common, calibrated, exact)]
        result.append(dict(N=int(pm["state_dimension"]), exact_tau_squared=em["precision_scale"],
                           compact_rms_tau_squared=cm["precision_scale"],
                           same_tau_covariance_error=relative(a, r),
                           separate_rms_covariance_error=relative(c, r),
                           normalization_change_norm_over_exact_covariance=float(np.linalg.norm(c-a)/np.linalg.norm(r))))
    return result


def refinement_check(summary):
    folder = BASE / "continuum-accuracy-cost"
    levels = sorted({r["refinement"] for r in summary if r["method"] == "supported"})
    reference = moments(folder / f"supported-n{levels[-1]}-rep0")
    reference_n = next(r["N"] for r in summary if r["refinement"] == levels[-1])
    return [dict(N=r["N"], reference_N=reference_n,
                 **errors(moments(folder / f"supported-n{r['refinement']}-rep0"), reference))
            for r in summary if r["method"] == "supported"]


def plot(pde, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
    h = np.array([1/r["refinement"] for r in pde])
    error = np.array([100*r["flux_relative_error"] for r in pde])
    axes[0].loglog(h, error, "o-", color="#2864b4", label="Independent PDE solve")
    axes[0].loglog(h, error[0]*(h/h[0])**2, "--", color="#888888", label=r"$h^2$ reference")
    axes[0].set(xlabel="Mesh spacing h", ylabel="Flux error against analytic solution (%)",
                title="Continuum forcing: spatial accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].xaxis.set_major_locator(FixedLocator([1/16, 1/8, 1/4]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"1/{round(1/x)}"))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].yaxis.set_major_locator(FixedLocator([.2, .5, 1, 2, 5]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    for ax, n in zip(axes[1:], (1206, 3032)):
        group = [r for r in summary if r["N"] == n and r["repetitions"] >= 3]
        for method in LABELS:
            for row in [r for r in group if r["method"] == method]:
                ax.errorbar(100*row["gate_error"], row["median_s"],
                            yerr=[[row["median_s"]-row["min_s"]], [row["max_s"]-row["median_s"]]],
                            fmt="s", capsize=3, color=COLORS[method], label=LABELS[method])
        spectral = sorted([r for r in group if r["method"].startswith("spectral-")], key=lambda r: int(r["method"][9:]))
        if spectral:
            ax.errorbar([100*r["gate_error"] for r in spectral], [r["median_s"] for r in spectral],
                        yerr=[[r["median_s"]-r["min_s"] for r in spectral], [r["max_s"]-r["median_s"] for r in spectral]],
                        fmt="o", capsize=3, color="#b15b12", label="Spectral GP (labels = rank)")
            for i, row in enumerate(spectral):
                ax.annotate(row["method"][9:], (100*row["gate_error"], row["median_s"]),
                            xytext=(7, -11 if row["method"] == "spectral-8" else 5), textcoords="offset points", fontsize=8, color="#994600")
        for tolerance in (1, 5):
            ax.axvline(tolerance, color="#aaaaaa", linestyle=":", linewidth=1)
        ax.set(xscale="symlog", yscale="log", xlabel="Maximum tested posterior error (%)",
               ylabel="Complete selected UQ setup (seconds)", title=f"Fixed sensors, N = {n:,}")
        ax.set_xscale("symlog", linthresh=.05)
        ax.set_xlim(-.008, 7)
        ax.set_ylim(min(r["min_s"] for r in group)*.78, max(r["max_s"] for r in group)*1.25)
        ax.set_xticks([0, .1, 1, 5], ["0", "0.1", "1", "5"])
        ax.yaxis.set_major_locator(FixedLocator([.5, 1, 2, 5, 10, 20]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(frameon=False, fontsize=7.5, loc="best")
    for ax in axes:
        ax.grid(True, alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("One-form PDE UQ: independent validation and cost at measured accuracy", fontsize=14)
    fig.text(.5, .025, "Fixed prior parameters; 6 physical sensors and 6 held-out fluxes. Means, covariances, marginal variances and 20 replacement datasets checked.\nTiming: median of 3 fresh runs after warmup, including geometry, assembly and eigensolves. PDE error and posterior approximation error are separate quantities.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .94))
    fig.savefig(BASE / "continuum-validation-and-cost.png", dpi=180)
    fig.savefig(BASE / "continuum-validation-and-cost.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    summary, audit = audit_uq(BASE / "continuum-accuracy-cost")
    contract, contract_audit = audit_uq(BASE / "continuum-contract-check")
    pde = audit_pde()
    table = tolerance_table(summary)
    normalization = audit_normalization()
    refinement = refinement_check(summary)
    for name, rows in [("continuum-summary.csv", summary), ("continuum-accuracy-audit.csv", audit + contract_audit),
                       ("continuum-pde-convergence.csv", pde), ("continuum-tolerance-cost.csv", table),
                       ("compact-normalization-ablation.csv", normalization),
                       ("continuum-refinement-check.csv", refinement)]:
        if rows:
            write_rows(BASE / name, rows)
    if args.plot:
        plot(pde, summary)
    print(f"Audited {len(audit) + len(contract_audit)} UQ runs, {len(pde)} independent PDE solves and {len(normalization)} normalization controls.")
    for row in summary:
        print(f"N={row['N']:5} {row['method']:14} {row['timing_kind']:12} time={row['median_s']:.4f}s gate={100*row['gate_error']:.4f}%")
    print("Fastest tested repeated configurations meeting each tolerance:")
    for row in table:
        print(row)


if __name__ == "__main__":
    main()
