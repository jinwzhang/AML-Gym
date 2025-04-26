from collections import Counter
from numpy import nan
from AMLGymCore.utils.functions import cell_to_discrete_number
from AMLGymCore.config import conf as config


def policy_check_for_traceable(config_df, record_keeper_df):
    actions = Counter(record_keeper_df['_valid_actions'].dropna().tolist())
    trade_value = {k: v * config.TRANSACTION_VALUE for k, v in actions.items()}
    config_df['_observable_tmp'] = config_df['trade_id'].apply(cell_to_discrete_number, pair=trade_value)
    config_df['_observable'] = config_df['_observable_tmp'] > config_df['SAR_threshold']
    config_df['_observable_id'] = config_df['trade_id'][config_df['_observable']==True]
    observable_id = set(config_df['_observable_id'].tolist())
    record_keeper_df['_traceable_actions'] = record_keeper_df['_valid_actions'].apply(lambda x: x if x in observable_id else nan)


def policy_check_uar_threashold(config_df, record_keeper_df):
    traceable_actions = Counter(record_keeper_df['_traceable_actions'].dropna().tolist())
    traceable_actions = {k: v * config.TRANSACTION_VALUE for k, v in traceable_actions.items()}

    config_df['_traceable_actions'] = config_df['trade_id'].apply(cell_to_discrete_number, pair=traceable_actions)

    config_df['_SAR'] = config_df['_traceable_actions'] > config_df['SAR_threshold']

    sar_amount = sum(config_df['_SAR'].dropna().tolist())

    trigger_sar = True if sar_amount > 0 else False
    # if trigger_sar:
    #     print(config_df[['trade_id', 'SAR', 'normal_amount']][config_df['traceable_actions'] > config_df['normal_amount']])
    #     import time
    #     time.sleep(10)
    return trigger_sar
