import copy
import numpy as np
from collections import Counter

import pandas as pd
import gymnasium as gym
from gymnasium import Env
from gymnasium.utils import seeding
import math
from AMLGymCore.utils.log import logger
from AMLGymCore.businesses.product import TradeType, trade_config_df
from AMLGymCore.businesses.reward_func import reward_function
from AMLGymCore.utils.functions import *
from AMLGymCore.businesses.aml_policy import policy_check_for_traceable, policy_check_uar_threashold


class Business(Env):

    def __init__(self, train_memo='default_', group=conf.GROUP, aml_policy=conf.AML_POLICY, aml_random_check=conf.AML_RAMDOM_CHECK):
        super().__init__()
        self.train_memo = train_memo
        self.group = group
        self.ingestion = conf.INGENSTION
        self.aml_policy = aml_policy
        self.aml_random_check = aml_random_check
        self.max_steps = conf.MAX_STEP

        self.log_name = 'BIZ-%s-' % str(train_memo) + dt.datetime.now().strftime('%Y%m%d_%H%M') + '.log'
        self.log = logger(self.log_name)
        self.log_path = conf.LOG_DIR + os.sep + self.log_name.split('.')[0] + os.sep
        self.N = 0

        self.record_keeper_df = pd.DataFrame()
        self.trade_config = trade_table_preprocess(trade_config_df, self.group).copy()
        self.trade_id_pair, self.product_name_pair = column_to_discrete_number(
            self.trade_config['trade_id']), column_to_discrete_number(self.trade_config['product_name'])
        self.trade_config['_limit_remains'] = self.trade_config['max_volume']
        self.trade_config['_observable'] = False
        self.trade_config['_current_trade_object'] = self.trade_config['trade_id'].apply(lambda x: TradeType(trade_id=x))
        self.trade_config['_current_trade_range'] = 0
        print(conf.LOG_DIR)
        self.trade_config.to_excel(conf.LOG_DIR + os.sep + 'trade_config_log_at_init.xlsx')
        self.record_keeper_df['_valid_actions'] = np.nan
        self.record_keeper_df['_invalid_actions'] = np.nan
        self.record_keeper_df['_reward'] = np.nan
        self.record_keeper_df['_paid'] = np.nan
        self.record_keeper_df['_returned'] = np.nan
        self.record_keeper_df['_assets'] = np.nan
        self.asset_holds_obs = self.trade_config['product_name'].copy().drop_duplicates()
        self.asset_holds = copy.deepcopy(self.ingestion)
        self.reward = 0
        self.rewards = []
        self.monetary_cost = 0
        self.n = 0
        self.arrested = False
        self.log.info(str(self.__dict__))
        self.max_reward, self.last_max_reward = 0, 0
        self.trade_id = ''
        self.path = []
        self.laundering_map = []
        self.record_laundering_map = []
        self.last_img = None
        self.terminated = False if not self.arrested else True
        self.truncated = self.n >= self.max_steps
        self.info = {'TimeLimit.truncated': self.truncated}
        self.validation = []

    def step(self, action):
        self.n += 1
        self.N += 1
        self._path_record()
        # refresh state
        self.trade_config['_current_trade_scope_value_range'] = self.trade_config['_current_trade_object'].apply(lambda x: self._trade_ability_eval(x.product_name))
        self.trade_config['_current_trade_scope_limit_range'] = (self.trade_config['_limit_remains'] / conf.TRANSACTION_VALUE).round(4)
        self.trade_config['_current_trade_scope_limit_range'].fillna(value=np.inf, inplace=True)
        self.trade_config['_current_trade_range'] = self.trade_config[['_current_trade_scope_value_range', '_current_trade_scope_limit_range']].min(axis=1).abs()
        try:
            self.trade_id = self.trade_config['trade_id'][self.trade_config['_trade_id_descre'] == action].values[0]  # Translate the action into TRADE_ID
        except:
            self.log.error(str(action) + '  ' + str(self.__dict__))
        if self.aml_policy:
            self._aml_policy()
        # if self.N > 1000:
        #     self.trade_config.to_excel(conf.LOG_DIR + os.sep + self.log_name.split('.')[
        #         0] + os.sep + 'trade_at_step%d_%d.xlsx' % (self.N, self.n))
        return self._do_trade(self.trade_id)
    @property
    def observation_value(self):
        asset_holds = {k: math.log(1/math.e + v/conf.TRANSACTION_VALUE) for k, v in self.asset_holds.items()}
        asset_holds_obs = self.asset_holds_obs.apply(cell_to_discrete_number, pair=asset_holds)
        self.trade_config['_temp_range'] = self.trade_config['_current_trade_range'].apply(lambda v: math.log(1/math.e + v))  # 20230829

        # 20230829 the ** 1/3 is for encourage agent to trade regardless of the asset on hand big or small
        trail1 = Counter(self.record_keeper_df['_valid_actions'].dropna().tolist())

        trail = {k: math.log(1/math.e + v) for k, v in trail1.items()}
        self.trade_config['_temp_trail'] = self.trade_config['trade_id'].apply(cell_to_discrete_number, pair=trail)
        trade_range_obs = self.trade_config['_temp_range'].to_numpy(na_value=0, dtype=np.float32).tolist()
        trade_trial_obs = self.trade_config['_temp_trail'].to_numpy(na_value=0, dtype=np.float32).tolist()
        asset_holds_obs = asset_holds_obs.to_numpy(na_value=0, dtype=np.float32).tolist()
        obs = copy.deepcopy(trade_range_obs)
        obs.extend(trade_trial_obs)
        obs.extend(asset_holds_obs)

        obs_space = np.array(obs)
        # if self.N % conf.MAX_STEP == 500:
        #     self.log.info('Current OBS: %d, %d' % (self.N, self.n))
        #     self.log.info(str(trade_range_obs))
        #     self.log.info(str(trade_trial_obs))
        #     self.log.info(str(asset_holds_obs))
        #     self.log.info(str(obs_space))

        return obs_space.flatten()

    @property
    def observation_space(self):
        space = gym.spaces.Box(low=-1, high=10, shape=(self.observation_value.shape[0],), dtype=np.float32)
        return space

    @property
    def action_space(self):
        act_space = self.trade_config['trade_id']
        act_space.fillna(value=0)
        act_space = act_space.to_numpy(na_value=0)
        act_space_n = gym.spaces.Discrete(act_space.shape[0])
        return act_space_n

    def reset(self, **kwargs):
        # Plot and log
        rewards = [t[1] for t in self.rewards]
        rewards.extend([self.reward, 1])
        self.max_reward = max(rewards)
        self.log.info(str(self.N) + ' RESET @n=%d reward=<%d> MAX=<%d>' % (self.n, self.reward, self.max_reward) + str(
            self.asset_holds))
        self.log.info('VALID: ' + str(Counter(self.record_keeper_df['_valid_actions'].dropna().tolist())))
        self.log.info('In-VALID: ' + str(Counter(self.record_keeper_df['_invalid_actions'].dropna().tolist())))
        if self.laundering_map != self.record_laundering_map and self.reward > (self.max_reward * 0.9):
            nx_plt(*self._render_data_prepare())  # Save path for good round.
        if rd.random() > 0.98 or self.reward >= self.max_reward > 200:
            self.record_keeper_df.to_excel(
                self.log_path + 'record_keeper_log_at_step%d_reward%d.xlsx' % (self.N, self.reward))
        self.rewards.append((self.N, self.reward))
        if self.N > 990000:
            with open(self.log_path + 'rewards.dat', 'w') as f:
                f.write(str(self.rewards))
        if self.N > 800000:
            success = self.asset_holds.get('income', 0) / self.ingestion.get('cash')
            self.validation.append(success)
            if len(self.validation) >= 10:
                success_rate = len([i for i in self.validation if i >= 0.6]) / len(self.validation)
                validation_words = 'Round info: %d, %d, success_rate=' % (
                    self.validation.count(True), len(self.validation)) + str(success_rate) + "MAX RATE: " + str(
                    max(self.validation)) + "MAX REWARD:" + str(self.max_reward)
                self.log.warn(validation_words)



        # Reset variable
        self.asset_holds = copy.deepcopy(self.ingestion)
        self.trade_config = trade_table_preprocess(trade_config_df, self.group).copy()
        self.trade_config['_limit_remains'] = self.trade_config['max_volume']
        self.trade_config['_current_trade_object'] = self.trade_config['trade_id'].apply(lambda x: TradeType(trade_id=x))
        self.trade_config['_current_trade_range'] = 0
        self.record_keeper_df['_valid_actions'] = np.nan
        self.record_keeper_df['_invalid_actions'] = np.nan
        self.record_keeper_df['_reward'] = np.nan
        self.record_keeper_df['_paid'] = np.nan
        self.record_keeper_df['_returned'] = np.nan
        self.record_keeper_df['_assets'] = np.nan
        self.asset_holds_obs = self.trade_config['product_name'].copy().drop_duplicates()
        self.reward = 0
        self.monetary_cost = 0
        self.n = 0
        self.arrested = False
        self.trade_id = ''
        self.path = []
        self.laundering_map = []
        self.record_laundering_map = []
        self.last_img = None
        self.terminated = False if not self.arrested else True
        self.truncated = self.n >= self.max_steps
        self.info = {'TimeLimit.truncated': self.truncated}
        return self.observation_value, self.info

    def render(self, mode="networkx"):
        show_latest_image(self.log_name)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        self.reset()
        self.N = 0

    @property
    def observation_shape(self):
        return self.observation_space.shape[0]

    def _do_trade(self, trade_id):
        trade_row = self.trade_config[self.trade_config['trade_id'] == trade_id].copy()

        current_trade_range = trade_row['_current_trade_range'].values[0]
        trade_object = trade_row['_current_trade_object'].values[0]
        current_trade_product = trade_object.product_name
        amount = min(1, sum(current_trade_product.values())) * conf.TRANSACTION_VALUE
        actual_amount = amount * min(current_trade_range, 1)
        actual_paid, exchange_return = {}, {}
        additional_asset_type = False
        if actual_amount > 5:
            self.record_keeper_df.at[self.n, '_valid_actions'] = trade_id
            actual_paid = {k: v * actual_amount for k, v in trade_object.product_name.items()}
            exchange_return = {k: v * actual_amount for k, v in trade_object.exchangeable_target.items()}
            self.trade_config.loc[self.trade_config['trade_id'] == trade_id, '_limit_remains'] -= conf.TRANSACTION_VALUE
            for asset1, value1 in actual_paid.items():  # deduct the paid asset
                self.asset_holds[asset1] -= value1
            for asset2, value2 in exchange_return.items():   # add the return asset
                if asset2 in self.asset_holds:
                    self.asset_holds[asset2] += value2
                else:
                    additional_asset_type = True
                    self.asset_holds[asset2] = value2

        new_state = self.observation_value
        if actual_amount > 5:
            # self.asset_holds = {k: v for k, v in self.asset_holds.items() if v > 5}  # drop the small fraction of assets
            self.record_keeper_df.at[self.n, '_paid'] = str(actual_paid)
            self.record_keeper_df.at[self.n, '_returned'] = str(exchange_return)
            self.record_keeper_df.at[self.n, '_assets'] = str(self.asset_holds)
            self.record_keeper_df.at[self.n, '_current_trade_range'] = current_trade_range
            self.laundering_map.append((actual_paid, exchange_return))
        else:
            self.record_keeper_df.at[self.n, '_invalid_actions'] = trade_id
            self.record_keeper_df.at[self.n, '_paid'] = str(trade_object.product_name)
            self.record_keeper_df.at[self.n, '_assets'] = str(self.asset_holds)
            self.record_keeper_df.at[self.n, '_current_trade_range'] = str(current_trade_range)
        self.terminated = False if not self.arrested else True
        self.truncated = self.n >= self.max_steps
        self.info = {'TimeLimit.truncated': self.truncated}
        params = {'product': actual_paid, 'actual_amount': actual_amount, 'trade_object': trade_object,
                  'exchange_return': exchange_return, 'c_range': current_trade_range,
                  'done': self.terminated, 'additional_asset_type': additional_asset_type, 'arrested': self.arrested, 'n': self.n}
        reward = reward_function(params)
        params.pop('trade_object')
        self.reward += reward
        self.log.info('N:%d, n:%d ' % (self.N, self.n) + ' Action: %s REWARD:%.2f , total reward %.2f' % (trade_id, reward, self.reward) +
                      ' params:' + str(params))
        self.record_keeper_df.at[self.n, '_reward'] = reward
        self.record_keeper_df.at[self.n, 'N'] = self.N
        self.record_keeper_df.at[self.n, 'n'] = self.n
        return new_state, reward, self.terminated, self.truncated, self.info

    def _render_data_prepare(self):
        points_and_weights, info = {}, None
        if self.path:
            points_and_weights = combine_full_path(self.laundering_map)
            self.log.info('Render full path Counter:' + str(points_and_weights))
            total_cost = self.assets_remaining.get('cost', 0)
            income = self.assets_remaining.get('income', 0)
            rate = 100 * income/sum(self.ingestion.values())
            income_info = 'Cost:%.1f, Income:%.1f=Pass Rate:%.2f%s' % (total_cost, income, rate, '%')
            info = 'N:%d Points:' % self.N + str(self.max_reward) + ' Max Step:' + str(self.max_steps) + ' Valid CNT:' + str(len(self.path)) + income_info + ' Log Name:' + self.log_name
            self.log.info(info)
            if __name__ == '__main__':
                nx_plt(points_and_weights, info)
            else:
                if self.last_max_reward == self.max_reward:
                    info = '_' + info
                else:
                    self.last_max_reward = self.max_reward
        return points_and_weights, info

    def _path_record(self):
        if self.max_reward * 0.85 < self.reward:
            self.path = self.record_keeper_df['_valid_actions'].dropna().tolist()
            self.assets_remaining = copy.deepcopy(
                {k: v for k, v in self.asset_holds.items() if v > conf.TRANSACTION_VALUE / 2})
            self.record_laundering_map = copy.deepcopy(self.laundering_map)
        # if self.N % 500000 in (50000-self.max_steps, 50000) and self.n == int(self.max_steps * 2/3):
        #     self.trade_config.to_excel(self.log_path + 'trade_config_log_at_step%d.xlsx' % self.N)


    def _aml_policy(self):
        if self.aml_random_check:  # random check to avoid agent learn the rule.
            check_interval = rd.randint(1, 50)
        else:
            check_interval = 1
        if self.n % check_interval == 0:  # Policy Check Point
            policy_check_for_traceable(config_df=self.trade_config,
                                       record_keeper_df=self.record_keeper_df)  # AML_policy to take effect
            self.arrested = policy_check_uar_threashold(config_df=self.trade_config,
                                                        record_keeper_df=self.record_keeper_df)
        return self.arrested

    def _trade_ability_eval(self, product):
        if isinstance(product, str):
            ability = self.asset_holds.get(product, 0) / conf.TRANSACTION_VALUE
        elif isinstance(product, dict):
            abilities = []
            for prod, ratio in product.items():
                abilities.append(self.asset_holds.get(prod, 0) / (conf.TRANSACTION_VALUE * ratio))
            ability = min(abilities)
        else:
            raise AttributeError('Config Product Error %s' % str(product))
        return round(ability, 4)




if __name__ == '__main__':
    import random as rd
    biz = Business()
    biz.step(1)


