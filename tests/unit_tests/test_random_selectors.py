#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


from unittest import TestCase
from parameterized import parameterized_class
import pandas as pd
import numpy as np

from dprecs.random_selectors import (
        report_noisy_max, 
        randomized_response, 
        greedy, 
        get_bag_winner, 
        get_snm_exp_val, 
        get_rr_exp_val,
        select_smart,
        )

@parameterized_class(
    [
        {
            "eligible":np.array((1,1,1,1,1)),
            "greedy_winner":2
        },
        {
            "eligible":np.array((0,0,0,1,1)),
            "greedy_winner":3
        },
    ]
)
class TestRandomizedResponse(TestCase):
    def setUp(self):
        self.pub_scores =  np.array([0.1, 0.5, 0.3, 0.2, 0.01])
        self.pvt_scores = np.array([0.2, 0.45, 0.6, 0.2, 0.05])
 
    def test_get_bag_winner(self):
        computed_ranks = get_bag_winner(
            bag_scores=pd.Series(self.pub_scores),
            pvt_scores=pd.Series(self.pvt_scores),
            bag_cutoff=1.0,
            epsilon_dp=15,
            mode="rr",
            pub_scores=None,
        )
        expected_ranks = np.zeros(self.pub_scores.shape)
        expected_ranks[self.pvt_scores.argmax()]=1
        np.testing.assert_allclose(
            computed_ranks.to_numpy(),
            expected_ranks,
            )
       
    def test_randomized_response(self):
        chosen_idx = randomized_response(self.pvt_scores, self.eligible, epsilon_dp=12)
        np.testing.assert_equal(chosen_idx, self.greedy_winner)

    def test_greedy(self):
        chosen_idx = greedy(self.pvt_scores, self.eligible)
        np.testing.assert_equal(chosen_idx, self.greedy_winner)

    def test_get_rr_exp_val(self):
        computed_value = get_rr_exp_val(self.pvt_scores, eps=20)
        np.testing.assert_allclose( computed_value, self.pvt_scores.max())


@parameterized_class(
    [
        {
            "pub_scores": np.array([0.1, 0.5, 0.3, 0.2, 0.01]),
            "pvt_scores": np.array([0.2, 0.45, 0.6, 0.2, 0.05]),
            "eligible": np.array((1,1,1,1,1)),
            "noise_type": "gumbel",
            "sensitivity_mode": "scale",
            "greedy_winner": 2,
        },
        {
            "pub_scores": np.array([0.1, 0.5, 0.3, 0.2, 0.01]),
            "pvt_scores": np.array([0.2, 0.45, 0.6, 0.2, 0.05]),
            "eligible": np.array((1,1,1,1,1)),
            "noise_type": "expo",
            "sensitivity_mode": "scale",
            "greedy_winner": 2,
        },
        {
            "pub_scores": np.array([0.1, 0.5, 0.3, 0.2, 0.01]),
            "pvt_scores": np.array([0.2, 0.45, 0.6, 0.2, 0.05]),
            "eligible": np.array((1,1,1,1,1)),
            "noise_type": "laplace",
            "sensitivity_mode": "scale",
            "greedy_winner": 2,
        },
        {
            "pub_scores": np.array([0.1, 0.5, 0.3, 0.2, 0.01]),
            "pvt_scores": np.array([0.2, 0.45, 0.6, 0.2, 0.05]),
            "eligible": np.array((1,1,0,1,1)),
            "noise_type": "expo",
            "sensitivity_mode": "scale",
            "greedy_winner": 1,
        },
        {
            "pub_scores": np.array([0.1, 0.5, 0.3, 0.2, 0.01]),
            "pvt_scores": np.array([0.2, 0.45, 0.6, 0.2, 0.05]),
            "eligible": np.array((0,0,0,1,0)),
            "noise_type": "expo",
            "sensitivity_mode": "scale",
            "greedy_winner": 3,
        },
        {
            "pub_scores": np.array([0.1, 0.5, 0.3, 0.2, 0.01]),
            "pvt_scores": np.array([0.2, 0.45, 0.6, 0.2, 0.05]),
            "eligible": np.array((1,1,1,1,1)),
            "noise_type": "expo",
            "sensitivity_mode": "smart_clip",
            "greedy_winner": 2,
        },
        {
            "pub_scores": np.array([0.1]),
            "pvt_scores": np.array([0.21]),
            "eligible": np.array((1)),
            "noise_type": "expo",
            "sensitivity_mode": "scale",
            "greedy_winner": 0,
        },
    ]
)
class TestSelectNoisyMax(TestCase):
    def setUp(self):
        pass

    def test_get_bag_winner(self):
        computed_ranks = get_bag_winner(
            bag_scores=pd.Series(self.pub_scores),
            pvt_scores=pd.Series(self.pvt_scores),
            bag_cutoff=1.0,
            epsilon_dp=100,
            mode="rnmg",
            pub_scores=pd.Series(self.pub_scores),
            clip_bound=1,
        )
        expected_ranks = np.zeros(self.pub_scores.shape)
        expected_ranks[self.pvt_scores.argmax()]=1
        np.testing.assert_allclose(
            computed_ranks.to_numpy(),
            expected_ranks,
            )
    
    def test_report_noisy_max_high_epsilon(self):
        chosen_idx = report_noisy_max(
                self.pvt_scores, 
                eligible = self.eligible,
                pub_scores=self.pub_scores, 
                epsilon_dp=100, 
                sensitivity_mode=self.sensitivity_mode, 
                clip_bound = 1,
                dist=self.noise_type )
        np.testing.assert_equal(chosen_idx, self.greedy_winner)

    def test_get_snm_exp_val(self):
        computed_value = get_snm_exp_val(self.pvt_scores, self.pub_scores, eps=500, clip_bound=1)
        np.testing.assert_allclose( computed_value, self.pvt_scores.max())
    
    def test_get_snm_exp_val_overflow(self):
        computed_value = get_snm_exp_val(self.pvt_scores*1e4, self.pvt_scores*1e4, eps=500, clip_bound=1)
        np.testing.assert_allclose( computed_value, self.pvt_scores.max()*1e4)
@parameterized_class(
    [
        {
            "selector_fun": lambda x: randomized_response(
                np.array((2,1)), 
                eligible=np.array((1,1)), 
                epsilon_dp=1
                ),
            "rtol":0.1,
            "expected_pdf": np.array((np.exp(1)/(2-1+np.exp(1)), 1/ (2-1+np.exp(1)) ) ),
        },
        {
            "selector_fun": lambda x: randomized_response(
                np.array((2,1)), 
                eligible=np.array((1,0)), 
                epsilon_dp=1
                ),
            "rtol":0.1,
            "expected_pdf": np.array((1,0)),
        },
        {
            "selector_fun": lambda x: report_noisy_max(
                np.array((2,1)), 
                eligible=np.array((1,1)),
                pub_scores = np.array((2,1)),
                sensitivity_mode = "smart_clip",
                dist="gumbel",
                clip_bound = 0.5,
                epsilon_dp=0.5
                ),
            "rtol": 0.1,
            "expected_pdf": np.exp(np.array((2,1))*0.5/2/1) / np.exp(np.array((2,1))*0.5/2/1).sum(),
        },
        {
            "selector_fun": lambda x: select_smart(
                np.array((2,1)), 
                eligible=np.array((1,1)),
                pub_scores = np.array((2,1)),
                sensitivity_mode = "clip",
                dist="expo",
                clip_bound = 0.5,
                epsilon_dp=20
                ),
            "rtol": 0.1,
            "expected_pdf": np.array((1,0)),
        },

    ]
)
class TestDistributions(TestCase):
    def setUp(self):
        self.iters = 5000

    def test_empirical_count(self):
        choice_counts = np.zeros(self.expected_pdf.shape)
        for i in range(self.iters):
            choice_counts[self.selector_fun()] += 1
        np.testing.assert_allclose(
                choice_counts/choice_counts.sum(), 
                self.expected_pdf, 
                rtol=self.rtol,
                err_msg=f"Expected PDF: {self.expected_pdf}"
                f"Observed pdf: {choice_counts/choice_counts.sum()}"
                )

