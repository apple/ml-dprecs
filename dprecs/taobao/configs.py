#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


from pyspark.sql.types import StructType, StringType, DoubleType, StructField
from collections import OrderedDict

TIME_STAMP = "time_stamp"
USER_ID = "userid"
AUCTION_ID = "synth_auction_id"
BID = "bid"
ITEM_PRICE = "price"
ADID = "adgroup_id"
SERVER_PCTR = "pclick_public"
CLICK = 'clk'
PROBA = 'proba'
BEST_PERSONALIZED_PCTR = "pclick_private"
PERSONALIZED_PCTR = "alpha_pclick_private"
RESERVE = "reserve"
NUM_FINALISTS = "num_finalists"

# Assume advertisers are willing to pay 1/100th of item price for an ad click
BID_MULT = 1e-2
# $0.01 is min item price found in the dataset
RESERVE_PRICE = 0.01 * BID_MULT
# Fraction of personalized pCTR in final on-device pCTR
ALPHA = 1.0
# Approximate mean CTR, to be used for naive/global pCTR model. Actual value does not affect results.
GLOBAL_PCTR = 0.05

# Where raw data is stored
INPUT_TABLE_NAME = "allegra_latimer.taobao_inputs"
# Where to save results
RESULTS_TABLE_NAME = "allegra_latimer.taobao_results"

############# Bag-of-contents configs to be tested ############################
GLOBAL_BOAS_CONFIGS = [
    OrderedDict(
        {"bag_scores": "eBPM", "bag_cutoff": 1, "pvt_scores": "pers_eBPM", "mode": "g"}
    ),
]

BOAS_CONFIGS = [
    OrderedDict(
        {"bag_scores": "eBPM", "bag_cutoff": 1, "pvt_scores": "pers_eBPM", "mode": "g"}
    ),
]
for eps in [0.1, 0.5, 1, 2, 3, 5, 7, 10, 15]:
    BOAS_CONFIGS.extend(
        [
            OrderedDict(
                {
                    "bag_scores": "eBPM",
                    "bag_cutoff": 1,
                    "pvt_scores": "pers_eBPM",
                    "epsilon_dp": eps,
                    "mode": "rr",
                }
            ),
            OrderedDict(
                {
                    "bag_scores": "eBPM",
                    "bag_cutoff": 1,
                    "pvt_scores": "pers_eBPM",
                    "epsilon_dp": eps,
                    "mode": "rnme",
                    "sensitivity_mode": "scale",
                    "pub_scores": "eBPM",
                }
            ),
            OrderedDict(
                {
                    "bag_scores": "eBPM",
                    "bag_cutoff": 1,
                    "pvt_scores": "pers_eBPM",
                    "epsilon_dp": eps,
                    "mode": "bst",
                    "pub_scores": "eBPM",
                    "clip_bound": 3,
                    "epsilon_frac": 0.5,
                }
            ),
            OrderedDict(
                {
                    "bag_scores": "eBPM",
                    "bag_cutoff": 1,
                    "pvt_scores": "pers_eBPM",
                    "epsilon_dp": eps,
                    "mode": "bst",
                    "pub_scores": "eBPM",
                    "clip_bound": 3,
                    "epsilon_frac": 0.25,
                }
            ),
        ]
    )
    for clip_bound in [1, 3, 10, 30, 100, 300]:
        BOAS_CONFIGS.append(
            OrderedDict(
                {
                    "bag_scores": "eBPM",
                    "bag_cutoff": 1,
                    "pvt_scores": "pers_eBPM",
                    "epsilon_dp": eps,
                    "mode": "rnme",
                    "sensitivity_mode": "smart_clip",
                    "pub_scores": "eBPM",
                    "clip_bound": clip_bound,
                }
            ),
        )
for bag_cutoff in [0.8, 0.6, 0.4, 0.2]:
    BOAS_CONFIGS.append(
        OrderedDict(
            {
                "bag_scores": "eBPM",
                "bag_cutoff": bag_cutoff,
                "pvt_scores": "pers_eBPM",
                "epsilon_dp": 5,
                "mode": "rr",
            }
        )
    )

############ Specifying expected schemas of returned dataframes #####################
AUCTION_COLS = [
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

BOAS_COLS = [
    f"{'_'.join([str(c) for c in config.values()])}_rank" for config in BOAS_CONFIGS
]
GLOBAL_BOAS_COLS = [
    f"{'_'.join([str(c) for c in config.values()])}_rank"
    for config in GLOBAL_BOAS_CONFIGS
]

# Ensure no duplicate configs were added
assert len(BOAS_COLS) == len(set(BOAS_COLS))

PERSONALIZED_AUCTION_COLS = [col + "_personalized" for col in AUCTION_COLS]
SERVER_AUCTION_COLS = [col + "_server" for col in AUCTION_COLS]
SERVER_AUCTION_COLS.extend([col.replace(".", "o") + "_server" for col in BOAS_COLS])
GLOBAL_AUCTION_COLS = [col + "_global" for col in AUCTION_COLS]
GLOBAL_AUCTION_COLS.extend(
    [col.replace(".", "o") + "_global" for col in GLOBAL_BOAS_COLS]
)

# Final schema for pandas UDF
FINAL_SCHEMA = StructType(
    [
        StructField(AUCTION_ID, StringType()),
        StructField(ADID, StringType()),
        StructField(CLICK, DoubleType()),
        StructField(PROBA, DoubleType()),
        StructField(BID, DoubleType()),
        StructField(RESERVE, DoubleType()),
        StructField(SERVER_PCTR, DoubleType()),
        StructField(BEST_PERSONALIZED_PCTR, DoubleType()),
        StructField(PERSONALIZED_PCTR, DoubleType()),
        StructField(NUM_FINALISTS, DoubleType()),
        *[StructField(c, DoubleType()) for c in PERSONALIZED_AUCTION_COLS],
        *[StructField(c, DoubleType()) for c in SERVER_AUCTION_COLS],
        *[StructField(c, DoubleType()) for c in GLOBAL_AUCTION_COLS],
    ]
)
