#!/usr/bin/env python3
"""Compare fixed-parameter PDE UQ runtime and error on the same output maps."""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from importlib.util import module_from_spec, spec_from_file_location
spec = spec_from_file_location("development_analysis", Path(__file__).with_name("analyze-one-form-pde-development.py"))
analysis = module_from_spec(spec)
spec.loader.exec_module(analysis)
BASE = analysis.BASE


def main():
    folder = BASE / "fixed-parameter-scaling"
    rows = [r for r in analysis.records(folder / "timings.csv") if r["status"] == "ok" and r["warmup"] == "False"]
    levels = sorted({int(r["refinement"]) for r in rows})
    fig, axes = plt.subplots(1, len(levels), figsize=(6.5 * len(levels), 4.8), squeeze=False)
    summary = []
    for level, ax in zip(levels, axes[0]):
        reference = analysis.covariance(folder / f"supported-n{level}-rep0/surface_covariance.csv")
        reference_mean = [float(r["mean"]) for r in analysis.records(folder / f"supported-n{level}-rep0/surface.csv")]
        methods = sorted({r["method"] for r in rows if int(r["refinement"]) == level}, key=lambda m: (1, int(m[9:])) if m.startswith("spectral-") else (0, list(analysis.LABELS).index(m)))
        spectral = []
        for method in methods:
            group = [r for r in rows if r["method"] == method and int(r["refinement"]) == level]
            candidate = analysis.covariance(folder / f"{method}-n{level}-rep0/surface_covariance.csv")
            candidate_mean = [float(r["mean"]) for r in analysis.records(folder / f"{method}-n{level}-rep0/surface.csv")]
            times = [float(r["selected_uq_total_s"]) for r in group]
            error = analysis.relative_frobenius(candidate, reference)
            variance_error = max(abs(candidate[i][i]/reference[i][i] - 1) for i in range(len(reference)))
            median = statistics.median(times)
            n = int(float(group[0]["state_dimension"]))
            row = dict(method=method,N=n,rank=int(method[9:]) if method.startswith("spectral-") else "",repetitions=len(group),
                       median_s=median,min_s=min(times),max_s=max(times),covariance_relative_frobenius=error,
                       maximum_relative_variance_error=variance_error,
                       surface_mean_relative_error=analysis.relative_frobenius([candidate_mean],[reference_mean]),
                       median_peak_GiB=statistics.median(float(r["process_peak_rss_bytes"])/2**30 for r in group))
            summary.append(row)
            if method.startswith("spectral-"):
                spectral.append(row)
                dy = 10 if row["rank"] == n else (-13 if error < .04 else 6)
                ax.annotate(str(row["rank"]),(100*error,median),xytext=(4,dy),textcoords="offset points",fontsize=8,color="#b15b12")
            else:
                ax.errorbar(100*error,median,yerr=[[median-min(times)],[max(times)-median]],marker="s",capsize=3,
                            color=analysis.COLORS[method],label=analysis.LABELS[method],linestyle="none")
        if spectral:
            ax.errorbar([100*r["covariance_relative_frobenius"] for r in spectral], [r["median_s"] for r in spectral],
                        yerr=[[r["median_s"]-r["min_s"] for r in spectral],[r["max_s"]-r["median_s"] for r in spectral]],
                        marker="o",color="#b15b12",label="Mixed spectral GP (labels = rank)",capsize=3,linewidth=1)
        ax.set(yscale="log",xlabel="Surface-flux covariance error (%)",ylabel="Time to selected UQ (seconds)",title=f"N = {n:,} active one-form unknowns")
        lower, upper = ax.get_ylim()
        ax.yaxis.set_major_locator(FixedLocator([t for t in [.5, 1, 2, 5, 10, 20, 50, 100] if lower <= t <= upper]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True,alpha=.2); ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=8,loc="best")
    fig.suptitle("One-form PDE UQ: runtime at measured uncertainty accuracy",fontsize=14)
    fig.text(.5,.03,"Fixed prior scales; identical PDE residual, noise metric, sensors and 12 outputs. All eigensolver costs included.\nMedian and range of 3 fresh runs after warmup. Calibration-inclusive timings are a separate workload.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.94))
    fig.savefig(BASE / "spectral-runtime-and-accuracy.png",dpi=180)
    fig.savefig(BASE / "spectral-runtime-and-accuracy.pdf")
    with (BASE / "spectral-summary.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    for row in summary: print(row)


if __name__ == "__main__":
    main()
