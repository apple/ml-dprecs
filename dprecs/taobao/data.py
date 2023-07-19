#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


def get_candidates(configs, spark):
    candidates_query = f"""
    SELECT 
        CAST(CONCAT({configs.USER_ID}, {configs.TIME_STAMP}) as string) as {configs.AUCTION_ID} 
        , CAST({configs.ADID} as string) AS {configs.ADID}
        , CAST({configs.CLICK} as float) AS {configs.CLICK}
        , CAST(1.0 as float) AS {configs.PROBA}
        , CAST({configs.ITEM_PRICE}*{configs.BID_MULT} as float) AS {configs.BID}
        , CAST({configs.RESERVE_PRICE} AS float) AS {configs.RESERVE}
        , CAST({configs.SERVER_PCTR} as float) AS {configs.SERVER_PCTR}
        , CAST({configs.BEST_PERSONALIZED_PCTR} as float) AS {configs.BEST_PERSONALIZED_PCTR}
        , CAST({configs.ALPHA} * {configs.BEST_PERSONALIZED_PCTR} + ( 1 - {configs.ALPHA} ) * {configs.SERVER_PCTR} as float) AS {configs.PERSONALIZED_PCTR}
    FROM {configs.INPUT_TABLE_NAME}
    WHERE ({configs.ITEM_PRICE}*{configs.BID_MULT}>={configs.RESERVE_PRICE})
    """
    return spark.sql(candidates_query)
