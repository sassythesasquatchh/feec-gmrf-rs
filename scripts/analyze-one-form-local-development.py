#!/usr/bin/env python3
"""Audit local one-form/curl UQ and bounded whole-state covariance distortion."""
import argparse
import csv
import importlib.util
from pathlib import Path
import statistics

import numpy as np

spec = importlib.util.spec_from_file_location("continuum", Path(__file__).with_name("analyze-one-form-continuum-development.py"))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
BASE = common.BASE
GROUPS = {"global_flux": slice(0, 12), "heldout_flux": slice(6, 12),
          "local_one_form": slice(12, 18), "local_curl": slice(18, 24)}


def moments(directory):
    rows = common.records(directory / "surface.csv")
    mean = np.array([float(r["mean"]) for r in rows])
    truth = np.array([float(r["truth"]) for r in rows])
    covariance = np.loadtxt(directory / "surface_covariance.csv", delimiter=",")
    replacements = np.loadtxt(directory / "replacement_surface_means.csv", delimiter=",")
    assert mean.shape == truth.shape == (24,)
    assert covariance.shape == (24, 24) and replacements.shape == (20, 24)
    assert all(np.isfinite(x).all() for x in (mean, truth, covariance, replacements))
    assert common.relative(covariance, covariance.T) < 1e-9
    eig = np.linalg.eigvalsh((covariance + covariance.T)/2)
    assert eig[0] >= -1e-9*eig[-1]
    return mean, truth, covariance, replacements


def errors(candidate, reference):
    m, truth, c, updates = candidate
    mr, tr, cr, ur = reference
    assert np.array_equal(truth, tr)
    result = {}
    for name, group in GROUPS.items():
        result[f"{name}_mean_error"] = common.relative(m[group], mr[group])
        result[f"{name}_covariance_error"] = common.relative(c[group, group], cr[group, group])
        result[f"{name}_replacement_mean_error"] = max(common.relative(a[group], b[group]) for a, b in zip(updates, ur))
    assert np.all(cr.diagonal() > 0)
    result["maximum_variance_error"] = float(np.max(np.abs(c.diagonal()/cr.diagonal()-1)))
    scale = np.sqrt(np.outer(cr.diagonal(), cr.diagonal()))
    result["standardized_joint_covariance_error"] = common.relative(c/scale, cr/scale)
    result["gate_error"] = max(result.values())
    return result


def audit(folder):
    path = folder / "timings.csv"
    if not path.exists():
        return [], []
    rows = common.records(path)
    assert all(r["status"] == "ok" for r in rows)
    summary, audit = [], []
    for level in sorted({int(r["refinement"]) for r in rows}):
        reference = moments(folder / f"supported-n{level}-rep0")
        methods = sorted({r["method"] for r in rows if int(r["refinement"]) == level})
        for method in methods:
            group = [r for r in rows if int(r["refinement"]) == level and r["method"] == method]
            measured = [r for r in group if r["warmup"] == "False"]
            checked = []
            for row in group:
                directory = folder / f"{method}-n{level}-rep{row['repeat']}"
                assert row["workload"] == "local" and float(row["local_outputs"]) == 1
                assert float(row["output_rows"]) == 24 and float(row["local_width"]) == .15
                assert float(row["precision_scale"]) == common.SCALE and float(row["trace_probes"]) == 0
                assert float(row["sensor_rows"]) == 6 and float(row["pde_sigma"]) == .1
                assert float(row["cached_update_maximum_relative_error"]) < 1e-8
                candidate = moments(directory)
                error = errors(candidate, reference)
                n = int(float(row["state_dimension"]))
                if method in {"supported", "gp", f"spectral-{n}"}:
                    assert error["gate_error"] < 1e-8, (directory, error)
                if method.startswith("spectral-"):
                    assert float(row["eigenbasis_relative_residual"]) < 1e-7
                    assert float(row["eigenbasis_mass_orthogonality_defect"]) < 1e-8
                    difference = reference[2] - candidate[2]
                    assert np.linalg.eigvalsh((difference+difference.T)/2)[0] >= -1e-8*np.linalg.norm(reference[2], 2)
                # Appending queries must preserve the previous twelve outputs.
                control = BASE / ("continuum-contract-check" if level == 4 else "continuum-accuracy-cost") / f"{method}-n{level}-rep0"
                if control.exists():
                    old = common.moments(control)
                    assert common.relative(candidate[0][:12], old[0]) < 1e-8
                    assert common.relative(candidate[2][:12, :12], old[2]) < 1e-8
                checked.append(error)
                audit.append(dict(directory=str(directory), **error))
            time_rows = measured or group
            times = [float(r["selected_uq_total_s"]) for r in time_rows]
            summary.append(dict(refinement=level, N=n, method=method, repetitions=len(measured),
                                timing_kind="repeated" if measured else "single pilot",
                                median_s=statistics.median(times), min_s=min(times), max_s=max(times),
                                median_peak_GiB=statistics.median(float(r["process_peak_rss_bytes"])/2**30 for r in time_rows),
                                **{key: max(e[key] for e in checked) for key in checked[0]}))
    return summary, audit


def bounds():
    folder = BASE / "local-precision-bounds-raw"
    if not (folder / "timings.csv").exists():
        return []
    result = []
    for level in (4, 6):
        compact = folder / f"compact-n{level}-rep0"
        exact = folder / f"supported-n{level}-rep0"
        if not (exact / "diagnostic_state_covariance.csv").exists():
            continue
        covariances = []
        raw_covariances = []
        skew = []
        for directory in (compact, exact):
            parameters = {r["quantity"]: float(r["value"]) for r in common.records(directory / "metrics.csv")}
            assert parameters["precision_scale"] == common.SCALE
            assert parameters["local_outputs"] == 1 and parameters["sensor_rows"] == 6
            covariance = np.loadtxt(directory / "diagnostic_state_covariance.csv", delimiter=",")
            output = np.loadtxt(directory / "diagnostic_output_map.csv", delimiter=",")
            selected = moments(directory)[2]
            assert np.isfinite(covariance).all() and np.isfinite(output).all()
            # Use explicit contractions as an independent non-BLAS audit path,
            # with floating-point exceptions and finite-value checks enabled.
            projected = np.einsum("ik,kj->ij", output, covariance, optimize=False)
            selected_check = np.einsum("ik,jk->ij", projected, output, optimize=False)
            assert common.relative(selected_check, selected) < 1e-8
            assert common.relative(covariance, covariance.T) < 1e-8
            maximum_skew = float(np.max(np.abs(covariance-covariance.T))/np.max(np.abs(covariance)))
            assert maximum_skew < 1e-8
            skew.append(maximum_skew)
            raw_covariances.append(covariance)
            covariances.append((covariance+covariance.T)/2)
        approximate, reference = covariances
        # Symmetric definite generalized problem via Cholesky congruence;
        # linear solves are used throughout, with no explicit inverse.
        lower = np.linalg.cholesky(reference)
        left = np.linalg.solve(lower, approximate)
        whitened = np.linalg.solve(lower, left.T).T
        assert common.relative(whitened, whitened.T) < 1e-8
        eigenvalues, orthogonal = np.linalg.eigh((whitened+whitened.T)/2)
        vectors = np.linalg.solve(lower.T, orthogonal[:, [0, -1]])
        assert eigenvalues[0] > 0
        residuals = []
        witness_variances = []
        for column, i in enumerate((0, len(eigenvalues)-1)):
            a = np.einsum("ij,j->i", raw_covariances[0], vectors[:, column], optimize=False)
            exact_action = np.einsum("ij,j->i", raw_covariances[1], vectors[:, column], optimize=False)
            b = eigenvalues[i] * exact_action
            rayleigh = np.einsum("i,i->", vectors[:, column], a, optimize=False) / np.einsum("i,i->", vectors[:, column], exact_action, optimize=False)
            assert abs(rayleigh/eigenvalues[i]-1) < 1e-8
            witness_variances.append((float(np.einsum("i,i->", vectors[:, column], exact_action, optimize=False)),
                                      float(np.einsum("i,i->", vectors[:, column], a, optimize=False))))
            residuals.append(float(np.linalg.norm(a-b)/(np.linalg.norm(a)+np.linalg.norm(b))))
        assert max(residuals) < 1e-8
        common.write_rows(BASE / f"local-variance-witness-N{len(eigenvalues)}.csv",
                          [dict(active_one_form_index=i, minimum_ratio_weight=vectors[i, 0], maximum_ratio_weight=vectors[i, 1])
                           for i in range(len(eigenvalues))])
        result.append(dict(N=len(eigenvalues), variance_ratio_min=eigenvalues[0],
                           variance_ratio_max=eigenvalues[-1],
                           precision_ratio_min=1/eigenvalues[-1], precision_ratio_max=1/eigenvalues[0],
                           worst_relative_variance_distortion=max(abs(eigenvalues[0]-1), abs(eigenvalues[-1]-1)),
                           maximum_relative_entry_skew=max(skew),
                           exact_variance_of_maximum_witness=witness_variances[1][0],
                           compact_variance_of_maximum_witness=witness_variances[1][1],
                           maximum_extreme_eigenpair_residual=max(residuals)))
    return result


def plot_comparison(local_table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    global_table = common.records(BASE / "continuum-tolerance-cost.csv")
    curves = [("supported", .01, "Exact supported GMRF", "#2864b4", "s"),
              ("compact", .01, "Compact GMRF, 1%", "#087f73", "D"),
              ("gp", .01, "Exact covariance GP", "#8b50a8", "x"),
              ("spectral", .01, "Spectral, 1%", "#b15b12", "o"),
              ("spectral", .05, "Spectral, 5%", "#c89c53", "v")]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1))
    for row_index, (title, table) in enumerate([("Global surface fluxes", global_table), ("Global fluxes + local one-form and curl averages", local_table)]):
        summary_path = BASE / ("continuum-summary.csv" if row_index == 0 else "local-summary.csv")
        details = {(int(r["N"]), r["method"]): r for r in common.records(summary_path)}
        for family, tolerance, label, color, marker in curves:
            selected = sorted([r for r in table if r["family"] == family and float(r["tolerance"]) == tolerance], key=lambda r: int(r["N"]))
            if not selected:
                continue
            for col, field in enumerate(("median_s", "median_peak_GiB")):
                error = None
                if col == 0:
                    runs = [details[(int(r["N"]), r["method"])] for r in selected]
                    error = [[float(r["median_s"])-float(r["min_s"]) for r in runs],
                             [float(r["max_s"])-float(r["median_s"]) for r in runs]]
                axes[row_index, col].errorbar([int(r["N"]) for r in selected], [float(r[field]) for r in selected],
                                             yerr=error, capsize=3, marker=marker, color=color, label=label, linewidth=1.5, markersize=5)
        ticks = sorted({int(r["N"]) for r in table})
        for col, ylabel in enumerate(("Complete selected UQ (seconds)", "Recorded process peak memory (GiB)")):
            ax = axes[row_index, col]
            ax.set(xscale="log", yscale="log", xlabel="Active one-form unknowns", ylabel=ylabel, title=title)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_major_locator(FixedLocator(([1, 2, 5, 10, 20, 50, 100] if row_index == 0 else [3, 5, 10, 20, 50]) if col == 0 else [.05, .1, .2, .5, 1, 2, 5]))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .055), fontsize=9)
    fig.suptitle("One-form PDE UQ: time and memory at matched posterior accuracy", fontsize=14)
    fig.text(.5, .015, "Median of 3 fresh runs after warmup; geometry, output maps and eigensolves included. Fastest tested eligible spectral rank.\nCompact is absent from the local panels because it fails 5%; other missing points were not measured.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .95))
    fig.savefig(BASE / "one-form-matched-accuracy.png", dpi=180)
    fig.savefig(BASE / "one-form-matched-accuracy.pdf")


def pde_convergence():
    path = BASE / "local-pde-validation/timings.csv"
    if not path.exists():
        return []
    result = []
    previous = None
    for row in sorted(common.records(path), key=lambda r: int(r["refinement"])):
        assert row["status"] == "ok" and row["method"] == "pde"
        assert row["workload"] == "local" and float(row["local_width"]) == .15
        level = int(row["refinement"])
        error_a, error_b = [float(row[f"local_{kind}_relative_error"]) for kind in ("one_form", "curl")]
        order_a = np.log(previous[1]/error_a)/np.log(level/previous[0]) if previous else ""
        order_b = np.log(previous[2]/error_b)/np.log(level/previous[0]) if previous else ""
        if previous:
            assert 1.8 < order_a < 2.2 and 1.8 < order_b < 2.2
        result.append(dict(N=int(float(row["state_dimension"])), refinement=level,
                           local_one_form_relative_error=error_a, local_curl_relative_error=error_b,
                           local_one_form_order=order_a, local_curl_order=order_b))
        previous = level, error_a, error_b
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounds", action="store_true", help="Run the bounded dense diagnostic after timed jobs finish")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    summary, checked = audit(BASE / "local-accuracy-cost")
    _, contract = audit(BASE / "local-contract-check")
    table = common.tolerance_table(summary)
    pde = pde_convergence()
    for name, rows in [("local-summary.csv", summary), ("local-accuracy-audit.csv", checked+contract),
                       ("local-tolerance-cost.csv", table), ("local-pde-convergence.csv", pde)]:
        if rows:
            common.write_rows(BASE / name, rows)
    print(f"Audited {len(checked)+len(contract)} local-output runs.")
    for r in summary:
        print(f"N={r['N']:5} {r['method']:14} {r['timing_kind']:12} time={r['median_s']:.4f}s gate={100*r['gate_error']:.3f}% A_cov={100*r['local_one_form_covariance_error']:.3f}% curl_cov={100*r['local_curl_covariance_error']:.3f}%")
    if args.bounds:
        with np.errstate(all="raise"):
            rows = bounds()
        if rows:
            common.write_rows(BASE / "local-precision-bounds.csv", rows)
        for row in rows:
            print(row)
    if args.plot:
        plot_comparison(table)


if __name__ == "__main__":
    main()
