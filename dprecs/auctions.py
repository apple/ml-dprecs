#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
import pandas as pd
from dprecs.random_selectors import get_bag_winner


def get_auction_outputs(est_pCTRs, bids, reserve):
    """
    Calculates the outputs of an auction (i.e., per-candidate eBPM, eBPM rank, eRPM, and prices):
    Accepts:
        estimated pCTRs (e.g., the ones to use in the auction but possibly not the ones used to simulate user clicks)
        bids
        reserve price: Float that gives the price setter to the last ranked auction candidate
    Returns: List of outputs:
        eBPMs: Effective bid per thousand impressions
        eBPM_ranks: Rank by eBPM
        eRPMs: Effective revenue per thousand impression
        prices: Price charged if click observed
    """
    eBPMs = est_pCTRs * bids * 1000
    eBPM_ranks = eBPMs.rank(ascending=False, method="first").astype(int)
    eRPMs = (
        eBPMs.sort_values(ascending=False)
        .shift(-1, fill_value=reserve * est_pCTRs.loc[eBPMs.idxmin()] * 1000)
        .sort_index()
    )
    prices = eRPMs / est_pCTRs / 1000
    return [eBPMs, eBPM_ranks, eRPMs, prices]

def get_dr_metrics(prices, bids, best_pCTRs, taps, probas):
    CTR = best_pCTRs.where(
        taps.isna() | probas==0, (taps - best_pCTRs) / probas + best_pCTRs
    )
    surplus = (bids - prices) * CTR
    revenue = prices * CTR * 1000
    return [CTR, surplus, revenue]

def get_dm_metrics(prices, bids, pCTRs):
    """
    Given a set of prices, bids and pCTRs, estimate the KPIs
    using the direct method (i.e., assuming click probabilities
    given by the pClick model).
    :param prices: pd.Series of per-click prices per candidate
    :param bids: pd.Series of per-click bids per candidate.
    :params pCTRs: pd.Series of probabilities of click per candidate.
    :returns: CTR, surplus, eRPM (expected revenue per mille)
    """
    CTR = pCTRs
    surplus = (bids - prices) * CTR
    eRPM = prices * CTR * 1000
    return [CTR, surplus, eRPM]


def get_bag_ranks(bag_configs, metric_map):
    """
    Given a set of configs for the bag of contents selection algorithm,
    run the auction for each of these settings and return the
    winner ranks for each auction
    :param bag_configs: List of config dicts
    :param metric_map: Dict mapping score names to score pd.Series
    :returns: List of one-hot pd.Series where the 1 value is the index of the winner.
    """
    bag_ranks = []
    for bag_config in bag_configs:
        args = {}
        for arg in bag_config:
            # Special keys that must be mapped to appropriate pd.Series
            if arg in ["bag_scores", "pvt_scores", "pub_scores"]:
                args[arg] = metric_map[bag_config[arg]]
            else:
                args[arg] = bag_config[arg]
        bag_rank = get_bag_winner(**args)
        bag_ranks.append(bag_rank)
    return bag_ranks


def run_paper_auction(est_pCTRs, bids, best_pCTRs, reserve, taps, probas, bag_configs={}):
    """
    :param est_pCTRs: The pCTR scores (pd.Series) to be used in the server-side
        auction, eg ranking and pricing.
    :param bids: The advertisers' click-bids (pd.Series)
    :param best_pCTRs: The best estimate of the pCTR score to be used both for
        on-device selection and also for metric estimation.
    :param reserve: The minimum price at which to sell the ad slot (float)
    :param bag_configs: List of all bag-of-contents config settings to be tested
    :returns: List of pd.Series statistics of the auction.
    """
    [eBPMs, eBPM_rank, eRPMs, prices] = get_auction_outputs(
        est_pCTRs, bids, reserve
    )
    stats = [eBPMs, eBPM_rank, eRPMs, prices]
    stats.extend( get_dm_metrics(prices, bids, best_pCTRs) )
    stats.extend( get_dr_metrics(prices, bids, best_pCTRs, taps, probas) )
    metrics_map = {
        "eBPM": eBPMs,
        "pCTR": est_pCTRs,
        "pers_pCTR": best_pCTRs,
        "pers_eBPM": bids * best_pCTRs * 1000,
    }
    # Add one-hot rank Series denoting which item was chosen for each bag config
    stats.extend(get_bag_ranks(bag_configs, metrics_map))
    return stats


def run_auction_group(df, configs, auction_fun):
    """
    Runs all the auctions being considered, currently:
        1. Auction using the online pCTR model for server ranking/pricing
        2. Auction using the offline pCTR model for server ranking/pricing
        3. Auction using a naive, single-valued pCTR model for server ranking/pricing.

    Arguments:
        df: a spark dataframe for a single auction (rows = number of candidates)
        configs: configs module
        auction_fun: the auction function to run, e.g. run_paper_auction.
    Returns:
        spark dataframe with new columns/metrics for auctions that were run.
    """
    df[configs.NUM_FINALISTS] = df.shape[0]
    reserve = df[configs.RESERVE].iloc[0]

    # Run a full-information auction where the personalized pCTR value is available at the server
    df = df.join(
        pd.DataFrame(
            auction_fun(
                est_pCTRs=df[configs.PERSONALIZED_PCTR],
                bids=df[configs.BID],
                best_pCTRs=df[configs.PERSONALIZED_PCTR],
                reserve=reserve,
                taps=df[configs.CLICK],
                probas=df[configs.PROBA],
            ),
            columns=df.index,
            index=configs.PERSONALIZED_AUCTION_COLS,
        ).T
    )
    # Run auctions with only the non-perosnalized pCTR value available at the server
    # (i.e. approach proposed in the paper)
    df = df.join(
        pd.DataFrame(
            auction_fun(
                est_pCTRs=df[configs.SERVER_PCTR],
                bids=df[configs.BID],
                best_pCTRs=df[configs.PERSONALIZED_PCTR],
                reserve=reserve,
                taps=df[configs.CLICK],
                probas=df[configs.PROBA],
                bag_configs=configs.BOAS_CONFIGS,
            ),
            columns=df.index,
            index=configs.SERVER_AUCTION_COLS,
        ).T
    )
    # Run auctions with only a naive "global" pCTR value available at the server
    df = df.join(
        pd.DataFrame(
            auction_fun(
                est_pCTRs=pd.Series(
                    np.full(df.shape[0], configs.GLOBAL_PCTR), index=df.index
                ),
                bids=df[configs.BID],
                best_pCTRs=df[configs.PERSONALIZED_PCTR],
                reserve=reserve,
                taps=df[configs.CLICK],
                probas=df[configs.PROBA],
                bag_configs=configs.GLOBAL_BOAS_CONFIGS,
            ),
            columns=df.index,
            index=configs.GLOBAL_AUCTION_COLS,
        ).T
    )
    return df
