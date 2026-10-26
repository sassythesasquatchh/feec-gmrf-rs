#!/usr/bin/env python3
"""Summarize the current one-form PDE development measurements and draw the result."""
import csv
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("out/c51-one-form-pde")
COLORS = {"compact": "#087f73", "supported": "#2864b4", "gp": "#8b50a8"}
LABELS = {"compact": "Compact projected GMRF", "supported": "Exact supported GMRF", "gp": "Exact covariance-form GP"}


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def covariance(path):
    with path.open() as stream:
        return [[float(x) for x in row] for row in csv.reader(stream)]


def relative_frobenius(candidate, reference):
    return math.sqrt(sum((a-b)**2 for x,y in zip(candidate,reference) for a,b in zip(x,y)) /
                     sum(x*x for row in reference for x in row))


def main():
    all_rows = []
    for directory in ["selected-scaling", "sparse-selected-repeated", "supported-reference-extension"]:
        path = BASE / directory / "timings.csv"
        if path.exists():
            for row in records(path):
                if row["status"] == "ok":
                    row["directory"] = directory
                    all_rows.append(row)
    rows = [r for r in all_rows if r["warmup"] == "False"]
    state_sizes = {int(r["refinement"]): int(float(r["state_dimension"])) for r in all_rows}
    summary = []
    accuracy = []
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1.5, 1]})
    for method in LABELS:
        n_values, medians, lows, highs = [], [], [], []
        for level in sorted(state_sizes):
            group = [r for r in rows if r["method"] == method and int(r["refinement"]) == level]
            if not group:
                continue
            times = [float(r["selected_uq_total_s"]) for r in group]
            n = int(float(group[0]["state_dimension"]))
            median = statistics.median(times)
            n_values.append(n); medians.append(median); lows.append(median-min(times)); highs.append(max(times)-median)
            summary.append({"method":method,"N":n,"repetitions":len(times),"median_s":median,"min_s":min(times),"max_s":max(times),
                            "median_peak_GiB":statistics.median(float(r["process_peak_rss_bytes"])/2**30 for r in group)})
        axes[0].errorbar(n_values,medians,yerr=[lows,highs],color=COLORS[method],marker="o",capsize=3,label=LABELS[method],linewidth=1.8)
    pilots = [r for r in all_rows if r["warmup"] == "True" and
              not any(x["method"] == r["method"] and x["refinement"] == r["refinement"] for x in rows)]
    for directory in ["sparse-field-pilot", "sparse-selected-pilot"]:
        path = BASE / directory / "timings.csv"
        if path.exists():
            pilots += [r for r in records(path) if r["status"] == "ok" and float(r["state_dimension"]) > 3032 and
                       not any(x["method"] == r["method"] and x["refinement"] == r["refinement"] for x in all_rows)]
    if pilots:
        for method in LABELS:
            group = sorted([r for r in pilots if r["method"] == method], key=lambda r: float(r["state_dimension"]))
            if group:
                axes[0].plot([float(r["state_dimension"]) for r in group], [float(r["selected_uq_total_s"]) for r in group],
                             color=COLORS[method],marker="o",markerfacecolor="white",linestyle="none",label=f"{method}: single runs")
    for level in sorted(state_sizes):
        by_method = {r["method"]: BASE / r["directory"] / f"{r['method']}-n{level}-rep0" for r in all_rows if int(r["refinement"]) == level and r["warmup"] == "True"}
        if not {"supported", "compact"}.issubset(by_method):
            continue
        exact = covariance(by_method["supported"] / "surface_covariance.csv")
        compact = covariance(by_method["compact"] / "surface_covariance.csv")
        exact_mean = [float(r["mean"]) for r in records(by_method["supported"] / "surface.csv")]
        compact_mean = [float(r["mean"]) for r in records(by_method["compact"] / "surface.csv")]
        n = state_sizes[level]
        accuracy.append({"N":n,"compact_covariance_relative_frobenius":relative_frobenius(compact,exact),
                         "compact_maximum_relative_variance_error":max(abs(compact[i][i]/exact[i][i]-1) for i in range(len(exact))),
                         "compact_surface_mean_relative_error":relative_frobenius([compact_mean],[exact_mean]),
                         "gp_supported_relative_frobenius":relative_frobenius(covariance(by_method["gp"] / "surface_covariance.csv"),exact) if "gp" in by_method else ""})
    axes[1].plot([r["N"] for r in accuracy],[100*r["compact_covariance_relative_frobenius"] for r in accuracy],"o-",color=COLORS["compact"],label="Full covariance (Frobenius)")
    axes[1].plot([r["N"] for r in accuracy],[100*r["compact_maximum_relative_variance_error"] for r in accuracy],"s--",color="#b16829",label="Largest marginal-variance error")
    axes[0].set(xscale="log",yscale="log",xlabel="Active one-form unknowns N",ylabel="Time to selected UQ (seconds)",title="Complete setup + mean + 12-flux covariance")
    axes[1].set(xscale="log",xlabel="Active one-form unknowns N",ylabel="Error relative to exact supported law (%)",title="Compact approximation: surface-flux errors",ylim=(0,max(20, 110*max(r["compact_maximum_relative_variance_error"] for r in accuracy))))
    for ax in axes:
        ax.grid(True,which="major",alpha=.2); ax.legend(frameon=False,fontsize=8,loc="upper left")
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("One-form PDE uncertainty quantification — development results",fontsize=14,y=.99)
    fig.text(.5,.035,"3D Whitney 1-forms · weak PDE residual + flux data · RMS normalization · Apple M4 Pro, one thread\nFilled points: median and range of fresh runs after warmup. Open points: single runs. Covariance baseline is the implemented exact evaluator.",ha="center",fontsize=8,color="#444444")
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(BASE / "runtime-and-accuracy.png",dpi=180)
    fig.savefig(BASE / "runtime-and-accuracy.pdf")
    for name,data in [("selected-summary.csv",summary),("selected-accuracy.csv",accuracy)]:
        with (BASE/name).open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(BASE / "runtime-and-accuracy.png")
    for row in summary: print(row)
    for row in accuracy: print(row)


if __name__ == "__main__":
    main()
