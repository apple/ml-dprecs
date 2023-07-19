#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
import pandas as pd


def get_bag_winner(
    bag_scores,
    pvt_scores,
    bag_cutoff=1.0,
    epsilon_dp=1.0,
    mode="rr",
    pub_scores=None,
    clip_bound=1.0,
    sensitivity_mode="smart_clip",
    epsilon_frac=0.5,
):
    """
    Given a series of candidate's bag_scores (a non-private, server-known quantity, e.g. eBPM),
    and pvt_scores (private, on-device quantity, e.g. personalized eBPM),
    reduce the eligible candidate set via the bag_cutoff and
    select a winner from the final set using a randomized algorithm.
    1. Reduce candidate set to ads with bag_score within delta of max bag_score
    2. Select an ad using a randomized response mechanism parameterized by epsilon_dp.
    Arguments:
        :param bag_scores: pd.Series. Server scores used to decrease the bag size
        :param pvt_scores: pd.Series. Private scores to be DP protected
        :param bag_cutoff: float. Threshold for bag scores to determine which candidates are sent to device
        :param epsilon_dp: float. Level of epsilon DP privacy
        :param mode: str. Type of local DP selection to use
        :param noise_type: str. If mode is "select_noisy_max," this is the distribution to sample from
        :param pub_scores: pd.Series. If mode is "select_noisy_max," these public scores will be used in clipping
        :param clip_bound: float. If mode is "select_noisy_max," this clipping bound (sensitivity) will be applied
    Returns:
        Array of length candidates with all zeros except winner marked as one.
    """
    # Determine set of eligible candidates based on cutoff
    eligible = (bag_scores >= bag_scores.max() * (1 - bag_cutoff)).astype(int)
    # Initialize bag ranks to be all zeros
    ranks_bag = pd.Series(np.zeros(bag_scores.shape[0]), index=bag_scores.index)
    if mode == "rr":
        chosen_idx = randomized_response(
            pvt_scores.values, eligible.values, epsilon_dp=epsilon_dp
        )
    elif mode == "rnmg":
        chosen_idx = report_noisy_max(
            pvt_scores.values,
            eligible.values,
            pub_scores.values,
            epsilon_dp=epsilon_dp,
            clip_bound=clip_bound,
            sensitivity_mode=sensitivity_mode,
            dist="gumbel",
        )
    elif mode == "rnme":
        chosen_idx = report_noisy_max(
            pvt_scores.values,
            eligible.values,
            pub_scores.values,
            epsilon_dp=epsilon_dp,
            clip_bound=clip_bound,
            sensitivity_mode=sensitivity_mode,
            dist="expo",
        )
    elif mode == "rnml":
        chosen_idx = report_noisy_max(
            pvt_scores.values,
            eligible.values,
            pub_scores.values,
            epsilon_dp=epsilon_dp,
            clip_bound=clip_bound,
            sensitivity_mode=sensitivity_mode,
            dist="laplace",
        )
    elif mode == "g":
        chosen_idx = greedy(pvt_scores.values, eligible.values)
    elif mode == "bst":
        chosen_idx = select_smart(
            pvt_scores=pvt_scores.values,
            eligible=eligible.values,
            pub_scores=pub_scores.values,
            epsilon_dp=epsilon_dp,
            clip_bound=clip_bound,
            use_pvt=True,
            epsilon_frac=epsilon_frac,
        )
    else:
        raise ValueError("Unknown mode specified.")
    ranks_bag.iloc[chosen_idx] = 1
    return ranks_bag


def report_noisy_max(
    pvt_scores,
    eligible,
    pub_scores=None,
    epsilon_dp=1,
    sensitivity_mode="smart_clip",
    clip_bound=1,
    dist="gumbel",
):
    """
    Report (select) noisy max selection mechanism. Chooses the ad with the largest noisy score.
    Arguments:
        pvt_scores: Private scores to be used in selection.
        eligible: Vector of same length as pvt_scores. Equal to 1 if candidate is "eligible", 0 otherwise.
            Eligibility can come eg from the server bag selection mechanism.
        pub_scores: Public scores, used if smart_clip mode is specified. Length must match pvt_scores.
        epsilon_dp: Privacy parameter, epsilon.
        sensitivity_mode: Mode to enforce sensitivity. Can be {clip, smart_clip, or scale}.
            clip: Naively clip scores to be at max equal to the clipping bound
            smart_clip: Clip private scores relative to the public scores.
            scale: Scale each score in a given request such that the max is 1 and min is 0.
        clip_bound: Clipping bound to enforce if sensitivity_mode = "clip" or "smart_clip".
        dist: Distribution from which to draw noise. Can be gumbel, laplace, or expo.
    Returns:
        One-hot series where chosen candidate's index = 1.
    """
    # If only one candidate is available
    if pvt_scores.shape[0] == 1:
        chosen_idx = 0
    else:
        if sensitivity_mode == "smart_clip":
            assert (
                pub_scores is not None
            ), "Must specify a valid pub_scores vector to use smart clipping"
            pvt_scores = clip_scores(pvt_scores, pub_scores, clip_bound)
            sensitivity = 2 * clip_bound
        elif sensitivity_mode == "clip":
            pvt_scores = np.minimum(pvt_scores, clip_bound)
            sensitivity = clip_bound
        elif sensitivity_mode == "scale":
            pvt_scores = scale_scores(pvt_scores, scale_max=1, scale_min=0)
            sensitivity = 1
        else:
            raise ValueError(
                "Unknown sensitivity_mode specified; choose from (smart_clip, clip, scale)"
            )
        noise_scale = 2 * sensitivity / epsilon_dp
        if dist == "gumbel":
            noise = np.random.gumbel(0, scale=noise_scale, size=pvt_scores.shape)
        elif dist == "expo":
            noise = np.random.exponential(scale=noise_scale, size=pvt_scores.shape)
        elif dist == "laplace":
            noise = np.random.laplace(0, scale=noise_scale, size=pvt_scores.shape)
        else:
            raise ValueError(
                "Unknown dist specified; choose from (gumbel, expo, laplace)"
            )
        # If the candidate is not eligible for selection, set its pvt_score to be an arbitrarily low value
        noise = np.where(eligible, noise, -noise_scale * 1e3)
        chosen_idx = (noise + pvt_scores).argmax()
    return chosen_idx


def randomized_response(pvt_scores, eligible, epsilon_dp=1):
    """
    Epsilon greedy mechanism for selection. Flips a coin to decide whether to choose the
    greedy candidate (highest scoring candidate). If the coin is tails, choose an option uniformly at random.
    Arguments:
        pvt_scores: Private scores to be used in selection.
        eligible: Vector of same length as pvt_scores. Equal to 1 if candidate is "eligible", 0 otherwise.
            Eligibility can come eg from the server bag selection mechanism.
        epsilon_dp: Privacy parameter, epsilon.
    Returns:
        One-hot series where chosen candidate's index = 1.
    """
    p_greedy = (np.exp(epsilon_dp) - 1) / (np.exp(epsilon_dp) + eligible.sum() - 1)
    if np.random.binomial(1, p_greedy) == 1:
        chosen_idx = (pvt_scores * eligible).argmax()
    else:
        chosen_idx = (np.random.rand(eligible.shape[0]) * eligible).argmax()
    return chosen_idx


def greedy(pvt_scores, eligible):
    """
    Purely greedy mechanism for selection.
    Arguments:
        pvt_scores: Private scores to be used in selection.
        eligible: Vector of same length as pvt_scores. Equal to 1 if candidate is "eligible", 0 otherwise.
            Eligibility can come eg from the server bag selection mechanism.
    Returns:
        One-hot series where chosen candidate's index = 1.
    """
    chosen_idx = (pvt_scores * eligible).argmax()
    return chosen_idx


def clip_scores(pvt_scores, pub_scores, clip_bound):
    """
    Returns "smart" clipped scores (using public info)
    """
    clipped_scores = np.maximum(
        np.minimum(pvt_scores, pub_scores + clip_bound), pub_scores - clip_bound
    )
    return clipped_scores


def scale_scores(pvt_scores, scale_max=1, scale_min=0):
    """
    Returns scaled scores via min-max scaling
    """
    if pvt_scores.max() == pvt_scores.min():
        scaled_scores = np.ones(pvt_scores.shape)
    else:
        scaled_scores = (pvt_scores - pvt_scores.min()) / (
            pvt_scores.max() - pvt_scores.min()
        )
    return scaled_scores


def get_rr_probas(scores, eps):
    """
    Returns the selection probability distribution of randomized response
    """
    n = scores.shape[0]
    probas = np.zeros((scores.shape))
    for p in range(n):
        if p == scores.argmax():
            probas[p] = np.exp(eps) / (n - 1 + np.exp(eps))
        else:
            probas[p] = 1 / (n - 1 + np.exp(eps))
    try:
        np.testing.assert_allclose(probas.sum(), 1)
    except AssertionError:
        probas = 1./n
    return probas


def get_expmech_probas(scores, sensitivity, eps):
    """
    Returns the selection probability distribution of the exponential mechanism
    """
    probas = np.exp(scores * eps / 2 / sensitivity) / np.sum(
        np.exp(scores * eps / 2 / sensitivity)
    )
    # For occasional overflow issues, assign full probability density to argmax
    if np.isnan(probas.sum()):
        probas = np.zeros(scores.shape)
        probas[scores.argmax()] = 1
    try:
        np.testing.assert_allclose(probas.sum(), 1)
    except AssertionError:
        probas = 1./scores.shape[0]
    return probas


def select_smart(
    pvt_scores,
    eligible,
    pub_scores,
    epsilon_dp,
    clip_bound,
    use_pvt=True,
    epsilon_frac=0.5,
    sensitivity_mode="smart_clip",
    dist="gumbel",
):
    """
    "Smart" selection mechanisms that combines SNM with clipping and RR. It chooses which mechanism
    to apply depending on whether the expected value of RR or exponential mechanism is higher.
    Arguments:
        pvt_scores: Private scores to be used in selection.
        eligible: Vector of same length as pvt_scores. Equal to 1 if candidate is "eligible", 0 otherwise.
            Eligibility can come eg from the server bag selection mechanism.
        pub_scores: Public scores, used if smart_clip mode is specified. Length must match pvt_scores.
        epsilon_dp: Privacy parameter, epsilon.
        clip_bound: Clipping bound to enforce if sensitivity_mode = "clip" or "smart_clip".
        use_pvt: Whether to use the pvt scores in estimating the expected value of each mechanism.
        epsilon_frac: Which fraction of epsilon budget to assign to initial mechanism selection.
        dist: Distribution from which to draw noise. Can be gumbel, laplace, or expo.
    Returns:
        One-hot series where chosen candidate's index = 1.
    """
    pvt_scores = pvt_scores * eligible
    pub_scores = pub_scores * eligible
    algo = "snm"
    if use_pvt:
        rr_val = get_rr_exp_val(pvt_scores, epsilon_dp * (1 - epsilon_frac))
        snm_val = get_snm_exp_val(
            pvt_scores, pub_scores, epsilon_dp * (1 - epsilon_frac), clip_bound
        )
        if (
            randomized_response(
                np.array((rr_val, snm_val)),
                eligible=np.array((1, 1)),
                epsilon_dp=epsilon_frac * epsilon_dp,
            )
            == 0
        ):
            algo = "rr"
    else:
        if get_rr_exp_val(
            pub_scores, epsilon_dp * (1 - epsilon_frac)
        ) >= get_snm_exp_val(
            pub_scores, pub_scores, epsilon_dp * (1 - epsilon_frac), clip_bound
        ):
            algo = "rr"
    if algo == "rr":
        chosen_idx = randomized_response(
            pvt_scores, eligible, epsilon_dp * (1 - epsilon_frac)
        )
    else:
        chosen_idx = report_noisy_max(
            pvt_scores,
            eligible=eligible,
            pub_scores=pub_scores,
            epsilon_dp=epsilon_dp * (1 - epsilon_frac),
            sensitivity_mode=sensitivity_mode,
            clip_bound=clip_bound,
            dist=dist,
        )
    return chosen_idx


def get_rr_exp_val(scores, eps):
    """
    Returns the expectation value of the randomized response mechanism
    """
    probas = get_rr_probas(scores, eps)
    return (probas * scores).sum()


def get_snm_exp_val(scores, pub_scores, eps, clip_bound):
    """
    Returns the expectation value of the exponential mechanism w/ smart clipping
    """
    clipped_scores = np.maximum(
        np.minimum(scores, pub_scores + clip_bound), pub_scores - clip_bound
    )
    probas = get_expmech_probas(clipped_scores, clip_bound * 2, eps)
    exp_val = (probas * scores).sum()
    return exp_val
