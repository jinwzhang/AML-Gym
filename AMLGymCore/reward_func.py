import time
from AMLGymCore.config.conf import TRANSACTION_VALUE

def reward_function(params):
    if params['arrested']:
        return -30
    if params['actual_amount'] == 0:  # for invalid action
        return -0.1
    #base_reward = params.get('return_value') / params.get('paid_value') * 0.3
    base_reward = 0.001 * params['n']
    income = params['exchange_return'].get('income', 0)
    reportable_income = params['exchange_return'].get('reportable profits', 0)
    bank_balance = params['exchange_return'].get('bank balance', 0)
    cost = params['exchange_return'].get('cost', 0)
    additional_asset_type = params['additional_asset_type']


    if income >= 0.7 * TRANSACTION_VALUE:
        add_point = 5
    elif income > 0.6 * TRANSACTION_VALUE:
        add_point = 3
    elif income > 0.1 * TRANSACTION_VALUE:
        add_point = 1
    else:
        add_point = 0

    if cost > 0.2 * TRANSACTION_VALUE:
        deduc_point = 2
    elif cost > 0.1 * TRANSACTION_VALUE:
        deduc_point = 1
    elif cost > 0.05 * TRANSACTION_VALUE:
        deduc_point = 0.5
    else:
        deduc_point = 0


    # add_point += 0.5 if reportable_income else 0
    #add_point += 1 * (bank_balance / 5000 - 0.5)

    add_point -= deduc_point

    trade_object = params['trade_object']
    if trade_object.group:
        add_point += 0.5
    if len(trade_object.product_name.keys()) >= 3:
        add_point += len(trade_object.product_name.keys()) * 0.5
    # elif params.get('exchange_return') in ('transfer', ):
    #     add_point = 0.5


    reward = base_reward + add_point
    # print(reward, params)
    # time.sleep(0.5)
    return reward


