#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import pyspark.sql.functions as F


def get_aggregates_pyspark(
    selectors,
    auction_data,
    estimators=["dm"],
    metrics=["eRPM", "surplus", "CTR"],
    special_metrics=[],
):
    """
    selectors: List of selector methods to get aggregates for
    auction_data: pyspark dataframe with results from auctions
    estimators: Estimators to get aggregates on, only "dm" (direct method) by default
    metrics: metrics to get mean result for
    special_metrics: metrics that require custom definitions, such as "cost per click"
    Returns: dict of aggregate statistics for each selector, eg:
        { "selector_1": {"eRPM": mean_erpm, "CTR": mean_ctr, "surplus": mean_surplus} }
    """
    aggs = {}
    for skey in selectors:
        pttr_model = skey.split("_")[-1]
        statistics = [
            F.mean(f"{m}_{est}_{pttr_model}") for est in estimators for m in metrics
        ]
        for sm in special_metrics:
            if sm == "CPC":
                statistics.extend(
                    [F.sum(f"eRPM_{est}_{pttr_model}") for est in estimators]
                )
                statistics.extend(
                    [F.sum(f"CTR_{est}_{pttr_model}") for est in estimators]
                )
        agg = auction_data.filter(f"{skey} = 1").select(statistics).collect()
        rowdict = agg[0].asDict()
        keys = [f"{m}_{est}" for est in estimators for m in metrics]
        aggs[skey] = {key: rowdict[f"avg({key}_{pttr_model})"] for key in keys}
        for sm in special_metrics:
            if sm == "CPC":
                for est in estimators:
                    aggs[skey][f"CPC_{est}"] = (
                        rowdict[f"sum(eRPM_{est}_{pttr_model})"]
                        / rowdict[f"sum(CTR_{est}_{pttr_model})"]
                        / 1000
                    )
    return aggs
