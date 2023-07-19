#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


from unittest import TestCase
from parameterized import parameterized_class
import pandas as pd
import numpy as np
from dprecs.auctions import (
    get_auction_outputs,
    get_dm_metrics,
    run_paper_auction,
    run_auction_group,
)


class MockConfigs:
    def __init__(self):
        auction_cols = [
            "eBPM",
            "eBPMrank",
            "eRPM",
            "price",
            "CTR_dm",
            "surplus_dm",
            "eRPM_dm",
            "CTR_dr",
            "surplus_dr",
            "eRPM_dr",
        ]
        self.GLOBAL_PCTR = 0.05
        self.SERVER_PCTR = "server_pCTRs"
        self.PERSONALIZED_PCTR = "pers_pCTRs"
        self.BID = "bids"
        self.RESERVE = "reserve"
        self.CLICK="clicks"
        self.PROBA="probas"
        self.GLOBAL_AUCTION_COLS = [col + "_global" for col in auction_cols]
        self.SERVER_AUCTION_COLS = [col + "_server" for col in auction_cols]
        self.PERSONALIZED_AUCTION_COLS = [col + "_personalized" for col in auction_cols]
        self.GLOBAL_BOAS_CONFIGS = {}
        self.BOAS_CONFIGS = {}
        self.NUM_FINALISTS = "num_finalists"


@parameterized_class(
    [
        {
            "server_pCTRs": pd.Series([0.1, 0.5, 0.3, 0.2, 0.01], name="server_pCTRs"),
            "clicks": pd.Series([0, 0, 1, 0, 0], name="clicks"),
            "probas": pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], name="probas"),
            "pers_pCTRs": pd.Series([0.2, 0.45, 0.6, 0.2, 0.05], name="pers_pCTRs"),
            "bids": pd.Series([10.0, 1.0, 6.0, 5.0, 15.0], name="bids"),
            "reserve": 1.0,
            "exp_eBPMs": pd.Series([1000, 500, 1800, 1000, 150]),
            "exp_eBPM_ranks": pd.Series([2, 4, 1, 3, 5]),
            "exp_eRPMs": pd.Series([1000, 150, 1000, 500, 1 * 1000 * 0.01]),
            "exp_prices": pd.Series([10, 0.3, 1 / 0.3, 2.5, 1]),
            "exp_CTR": pd.Series([0.2, 0.45, 0.6, 0.2, 0.05]),
            "exp_revenue": pd.Series([2000, 135, 2000, 500, 50]),
            "exp_surplus": pd.Series([0, 0.7 * 0.45, 1.6, 0.5, 0.7]),
            "bag_configs": [
                {
                    "bag_scores": "eBPM",
                    "bag_cutoff": 1,
                    "pvt_scores": "pers_eBPM",
                    "mode": "g",
                },
            ],
            "exp_bag_ranks": [[0, 0, 1, 0, 0]],
            "configs": MockConfigs(),
        },
    ]
)
class TestAuction(TestCase):
    def setUp(self):
        self.df = pd.concat([self.server_pCTRs, self.pers_pCTRs, self.bids, self.clicks, self.probas], axis=1)
        self.df[self.configs.RESERVE] = self.reserve

    def test_get_auction_outputs(self):
        exp_outputs = [
            self.exp_eBPMs,
            self.exp_eBPM_ranks,
            self.exp_eRPMs,
            self.exp_prices,
        ]
        computed_outputs = get_auction_outputs(
            self.server_pCTRs, self.bids, self.reserve
        )
        for i, output in enumerate(exp_outputs):
            np.testing.assert_allclose(output, computed_outputs[i])

    def test_get_dm_metrics(self):
        exp_metrics = [self.pers_pCTRs, self.exp_surplus, self.exp_revenue]
        computed_metrics = get_dm_metrics(self.exp_prices, self.bids, self.pers_pCTRs)
        for i, output in enumerate(exp_metrics):
            np.testing.assert_allclose(output, computed_metrics[i])

    def test_run_paper_auction(self):
        exp_stats = [
            self.exp_eBPMs,
            self.exp_eBPM_ranks,
            self.exp_eRPMs,
            self.exp_prices,
            self.exp_CTR,
            self.exp_surplus,
            self.exp_revenue,
        ]
        exp_stats.extend(self.exp_bag_ranks)
        computed_stats = run_paper_auction(
            self.server_pCTRs,
            self.bids,
            self.pers_pCTRs,
            self.reserve,
            self.clicks,
            self.probas,
            bag_configs=self.bag_configs,
        )
        for i, output in enumerate(exp_stats):
            np.testing.assert_allclose(output, computed_stats[i])

    def test_run_auction_group(self):
        exp_df_shape = (5, 37)
        computed_df = run_auction_group(self.df, self.configs, run_paper_auction)
        np.testing.assert_equal(exp_df_shape, computed_df.shape)
