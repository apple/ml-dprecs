#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import matplotlib.pyplot as plt
from itertools import cycle
import string


def plot_aggregates(
    aggregates,
    groups,
    baseline="eBPMrank_personalized",
    figsize=(12, 3),
    xticks=None,
    xlabel=None,
    title=None,
    savepath=None,
    estimator="dr",
    metrics=["CTR", "eRPM", "surplus"],
    rename_metrics=["CTR", "Revenue", "Surplus"],
    ylims={"eRPM": (-75, 150), "surplus": (-10, 20), "CTR": (-10, 20)},
):
    fig, axs = plt.subplots(1, len(metrics), figsize=figsize)
    linestyles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (1, 10)),
        (5, (10, 3)),
        (0, (5, 10)),
    ]
    if not xticks:
        xticks = range(len(next(iter(groups.values()))))

    for m, metric in enumerate(metrics):
        for s, sgkey in enumerate(groups):
            lifts = []
            for selector in groups[sgkey]:
                lifts.append(
                    (
                        aggregates[selector][f"{metric}_{estimator}"]
                        - aggregates[baseline][f"{metric}_{estimator}"]
                    )
                    / aggregates[baseline][f"{metric}_{estimator}"]
                    * 100
                )
            axs[m].plot(
                xticks,
                lifts,
                color="k",
                ls=linestyles[s],
                label=sgkey,
            )
        axs[m].grid(visible=True)
        axs[m].set_ylabel(f"{rename_metrics[m]} % lift")
        if xlabel:
            axs[m].set_xlabel(xlabel)
        axs[m].set_ylim(ylims[metric])
        if axs[m].get_legend_handles_labels()[1]:
            axs[m].legend()
        axs[m].text(
            -0.1,
            1.1,
            string.ascii_lowercase[m],
            transform=axs[m].transAxes,
            size=20,
            weight="bold",
        )

    plt.suptitle(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()
