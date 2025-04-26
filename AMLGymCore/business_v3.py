import copy
import datetime as dt
import os
from collections import Counter

import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import ray
from gym import Env
from gym.utils import seeding
from matplotlib.colors import Normalize
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['AR PL UMing CN']
mpl.rcParams['axes.unicode_minus'] = False
from AMLWorld.utils.log import logger
from AMLWorld.config import conf as config
from AMLWorld.businesses.product import TradeType
from AMLWorld.businesses.reward_func import reward_function
from AMLWorld.utils.functions import *
from AMLWorld.businesses.aml_policy import policy_check_for_traceable, policy_check_uar_threashold
business_config_df = pd.read_excel(config.ENV_CONFIG_EXCEL, sheet_name='business')
trade_config_df = pd.read_excel(config.ENV_CONFIG_EXCEL, sheet_name='trade')
trade_config_df = trade_config_df.sample(frac=1)  # 20230901 Add config rows sequence shuffle
trade_config_df.dropna(axis=0, subset='trade_id', inplace=True)
trade_config_df.drop_duplicates(subset='trade_id', inplace=True)
trade_config_df.fillna(method='ffill')
trade_config_df['trade_id'] = trade_config_df['trade_id'].apply(lambda s: str(s).lower().strip())


@ray.remote
class Business(Env):

    def __init__(self, business_id, business_number, ingestion: dict, aml_policy, aml_random_check=False, max_steps=600, amount=config.TRANSACTION_VALUE):
        super().__init__()
        self.business_id = business_id
        self.business_number = business_number
        self.ingestion = copy.deepcopy(ingestion)
        self.aml_policy = aml_policy
        self.aml_random_check = aml_random_check
        self.max_steps = max_steps
        self.amount = amount
        self.log_name = 'BIZ-%d-' % business_number + dt.datetime.now().strftime('%Y%m%d_%H%M') + '.log'
        self.log = logger(self.log_name)
        self.N = 0
        self.asset_holds = copy.deepcopy(ingestion)
        self.record_keeper_df = pd.DataFrame()
        self.attributes = business_config_df[business_config_df['business_number'] == business_number]
        self.attributes = self.attributes.to_dict('records')[0]
        self.business_name = self.attributes.get('business_name')
        self.business_type = self.attributes.get('business_type')
        self.business_trade_scope = string_to_list(self.attributes.get('trade_list'))

        #self.trade_config = trade_config_df[trade_config_df['trade_id'].isin(self.business_trade_scope)].copy()  # 20230827 filter out irrelevant trade_ids
        self.trade_config = trade_config_df.copy()

        self.trade_id_pair, self.product_name_pair = column_to_discrete_number(self.trade_config['trade_id']), column_to_discrete_number(self.trade_config['product_name'])

        self.trade_config['trade_id_descre'] = self.trade_config['trade_id'].apply(cell_to_discrete_number, pair=self.trade_id_pair)
        self.trade_config['product_name_descre'] = self.trade_config['product_name'].apply(cell_to_discrete_number, pair=self.product_name_pair)
        self.trade_config['limit_remains'] = self.trade_config['max_volume']
        self.trade_config['observable'] = False
        self.trade_config['current_trade_object'] = self.trade_config['trade_id'].apply(lambda x: TradeType(trade_id=x))
        self.trade_config['current_trade_range'] = 0
        self.trade_config.to_excel(r'/data/rl/AMLGym/AMLGymCore/exp_log/123.xlsx')
        self.record_keeper_df['valid_actions'] = np.nan
        self.record_keeper_df['invalid_actions'] = np.nan
        self.record_keeper_df['traceable_actions'] = np.nan
        self.record_keeper_df['untraceable_actions'] = np.nan

        self.asset_holds_obs = self.trade_config['product_name'].copy().drop_duplicates()
        self.reward = 0
        self.monetary_cost = 0
        self.n = 0
        self.arrested = False
        self.log.info(str(self.__dict__))
        self.max_reward, self.last_max_reward = 0, 0
        self.trade_id = ''
        self.render_path = []
        self.assets_remaining = {}
        delattr(self, 'attributes')

    @ray.method(num_returns=4)
    def step(self, action):

        self.n += 1
        self.N += 1
        # for logging

        if self.max_reward * 0.85 < self.reward:
            self.max_reward = max(self.reward, self.max_reward)
            self.render_path = self.record_keeper_df['valid_actions'].dropna().tolist()
            self.assets_remaining = copy.deepcopy({k: v for k, v in self.asset_holds.items() if v > config.TRANSACTION_VALUE/2})
            self.log.info('New good render_path:' + str(self.render_path))
        if self.aml_policy:
            if self.aml_random_check:  # random check to avoid agent learn the rule.
                check_interval = rd.randint(1, 30)
            else:
                check_interval = 1

            if self.n % check_interval == 0:  # Policy Check Point
                policy_check_for_traceable(config_df=self.trade_config, record_keeper_df=self.record_keeper_df)  # AML_policy to take affect
                self.arrested = policy_check_uar_threashold(config_df=self.trade_config, record_keeper_df=self.record_keeper_df)



        # refresh state
        on_hand_asset_type = self.asset_holds.keys()
        current_trade_id_scope = [trade_id for trade_id in self.business_trade_scope if (trade_id.split('-')[0] in on_hand_asset_type)]
        current_trade_scope_value_range = [self.asset_holds[trade_id.split('-')[0]] for trade_id in current_trade_id_scope]
        current_trade_scope_value_range_dict = {k: v for k, v in zip(current_trade_id_scope, current_trade_scope_value_range)}
        self.trade_config['current_trade_scope_value_range'] = self.trade_config['trade_id'].apply(cell_to_discrete_number, pair=current_trade_scope_value_range_dict)
        self.trade_config['current_trade_scope_limit_range'] = self.trade_config['current_trade_object'].values[0].max_volume
        self.trade_config['current_trade_scope_value_range'].fillna(value=0, inplace=True)
        self.trade_config['current_trade_scope_limit_range'].fillna(value=99999999999, inplace=True)
        self.trade_config['current_trade_range'] = self.trade_config[['current_trade_scope_value_range', 'current_trade_scope_limit_range']].min(axis=1)

        self.trade_id = self.trade_config['trade_id'][self.trade_config['trade_id_descre'] == action].values[0]  # Translate the action into TRADE_ID
        current_trade_range = self.trade_config[self.trade_config['trade_id'] == self.trade_id]['current_trade_range'].values[0]


        if current_trade_range > 0.1:
            amount = min(current_trade_range, self.amount)
        else:
            amount = 0
        if self.N % 50000 > 49000 and self.n == int(self.max_steps * 2/3):
            self.trade_config.to_excel(config.LOG_DIR + os.sep + 'trade_config_%d.xlsx' % self.N)
            self.record_keeper_df.to_excel(config.LOG_DIR + os.sep + 'record_keeper_%d.xlsx' % self.N)
        return self.do_trade(self.trade_id, amount)

    @ray.method(num_returns=1)
    def observation_space(self):
        asset_holds = {k: round(v / self.amount, 2)**0.5 for k, v in self.asset_holds.items()}  # 20230806 **0.5 for good performance
        asset_holds_obs = self.asset_holds_obs.apply(cell_to_discrete_number, pair=asset_holds)
        #obs_space['range'] = obs_space['range'] * (obs_space.index + 1)  # conbine feature  20230718
        self.trade_config['temp_range'] = self.trade_config['current_trade_range'].apply(lambda x: 1 if x > 0 else 0)  # 20230829
        # 20230829 the ** 1/3 is for encourage agent to trade regardless of the asset on hand big or small
        self.trade_config['temp_trail'] = self.trade_config['trade_id'].apply(cell_to_discrete_number,
                                            pair=Counter(self.record_keeper_df['valid_actions'].dropna().tolist()))
        trade_range_obs = self.trade_config['temp_range'].to_numpy(na_value=0, dtype=np.float32).tolist()
        trade_trial_obs = self.trade_config['temp_trail'].to_numpy(na_value=0, dtype=np.float32).tolist()
        asset_holds_obs = asset_holds_obs.to_numpy(na_value=0, dtype=np.float32).tolist()
        obs = copy.deepcopy(trade_range_obs)
        obs.extend(asset_holds_obs)
        obs.extend(trade_trial_obs)

        # print(len(obs), obs)
        obs_space = np.array(obs)

        return obs_space.flatten()

    @ray.method(num_returns=1)
    def action_space(self):
        act_space = self.trade_config['trade_id']
        act_space.fillna(value=0)
        act_space = act_space.to_numpy(na_value=0)

        act_space_n = gym.spaces.Discrete(act_space.shape[0])
        return act_space_n

    def reset(self, **kwargs):
        self.log.info(str(self.N) + ' RESET @n=%d reward=<%d> MAX=<%d>' % (self.n, self.reward, self.max_reward) + str(self.asset_holds))
        self.log.info('VALID: ' + str(Counter(self.record_keeper_df['valid_actions'].dropna().tolist())))
        self.log.info('In-VALID: ' + str(Counter(self.record_keeper_df['invalid_actions'].dropna().tolist())))
        self.log.info(str(self.N) + '\t' + str(self.asset_holds))
        self.asset_holds = copy.deepcopy(self.ingestion)

        # self.trade_config = trade_config_df[trade_config_df['trade_id'].isin(self.business_trade_scope)].copy()  # 20230827 filter out irrelevant trade_ids
        self.trade_config = trade_config_df.copy()

        self.trade_id_pair, self.product_name_pair = column_to_discrete_number(
            self.trade_config['trade_id']), column_to_discrete_number(self.trade_config['product_name'])
        self.trade_config['trade_id_descre'] = self.trade_config['trade_id'].apply(cell_to_discrete_number, pair=self.trade_id_pair)
        self.trade_config['product_name_descre'] = self.trade_config['product_name'].apply(cell_to_discrete_number, pair=self.product_name_pair)
        self.trade_config['limit_remains'] = self.trade_config['max_volume']
        self.trade_config['observable'] = False
        self.trade_config['current_trade_object'] = self.trade_config['trade_id'].apply(lambda x: TradeType(trade_id=x))
        self.trade_config['current_trade_range'] = 0
        self.asset_holds_obs = self.trade_config['product_name'].copy().drop_duplicates()
        self.reward = 0
        self.monetary_cost = 0
        self.n = 0
        self.record_keeper_df['valid_actions'] = np.nan
        self.record_keeper_df['invalid_actions'] = np.nan
        self.record_keeper_df['traceable_actions'] = np.nan
        self.record_keeper_df['untraceable_actions'] = np.nan
        return self.observation_space()

    def render(self, mode='networkx'):
        DG = nx.DiGraph()
        info = None
        if self.render_path:
            points_and_weights = Counter(self.render_path)
            self.log.info('Render full path:' + str(self.render_path))
            self.log.info('Render full path Counter:' + str(points_and_weights))

            total_cost_of_income = monetary_diff(self.ingestion, self.assets_remaining)
            income = self.assets_remaining.get('income', 0)
            rate = 100 * (total_cost_of_income-income)/total_cost_of_income if income else 100
            income_info = 'Cost:%.1f/Income:%.1f=LaunderingRate:%.2f%s' % (total_cost_of_income-income, income, rate, '%')
            info = 'Points:' + str(self.max_reward) + ' Max Step:' + str(self.max_steps) + ' Valid CNT:' + str(len(self.render_path)) + income_info + ' Log Name:' + self.log_name
            for k, v in points_and_weights.items():
                config_row = trade_config_df[trade_config_df['trade_id'] == k]
                out_node = config_row['product_name'].values[0].strip()
                in_node = config_row['exchangeable_target'].values[0].strip()
                DG.add_node(in_node)
                DG.add_node(out_node)
                DG.add_weighted_edges_from([(out_node, in_node, v)])
            if __name__ == '__main__':
                nx_plt(DG, info)
            else:
                if self.last_max_reward == self.max_reward:
                    info = '_' + info
                else:
                    self.last_max_reward = self.max_reward
        return DG, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        self.reset()
        self.N = 0

    @ray.method(num_returns=1)
    def observation_shape(self):
        return self.observation_space().shape[0]

    def counter_parties_search(self):
        pass

    @ray.method(num_returns=4)
    def do_trade(self, trade_id, amount):
        params = {'paid_value': 0, 'return_value': 0, 'exchange_return': '', 'done': False, 'arrested': self.arrested}

        if amount >= 500:
            self.record_keeper_df.at[self.n, 'valid_actions'] = trade_id
            trade_object = self.trade_config[self.trade_config['trade_id'] == trade_id]['current_trade_object'].values[0]
            paid_value, return_value = trade_object.trasaction(amount)
            self.monetary_cost += (paid_value - return_value)
            params['paid_value'] = paid_value
            params['return_value'] = return_value
            params['exchange_return'] = trade_object.exchangeable_target
            self.asset_holds[trade_object.product_name] -= paid_value
            self.asset_holds = {k: v for k, v in self.asset_holds.items() if v > 500}  # drop the small fraction of assets
            if trade_object.exchangeable_target in self.asset_holds:
                self.asset_holds[trade_object.exchangeable_target] += return_value
            else:
                self.asset_holds[trade_object.exchangeable_target] = return_value
            params.update(self.asset_holds)


        else:
            self.record_keeper_df.at[self.n, 'invalid_actions'] = trade_id



        reward = reward_function(params)
        self.reward += reward
        self.log.info('Action: %s REWARD:%.2f , total reward %.2f' % (trade_id, reward, self.reward) + ' params:' + str(params))
        done = False if (self.asset_holds and self.n < self.max_steps and not params['arrested']) else True
        params['done'] = done
        return self.observation_space(), reward, done, False


def mid_lable(xy, text, xytext):
    xmid = 2/3 * xy[0] + 1/3 + xytext[0]
    ymid = 2/3 * xy[1] + 1/3 * xytext[1]
    return [xmid, ymid]

def nx_plt(DG, info):
    if info:
        pos1 = nx.circular_layout(DG)
        nx.draw_networkx_nodes(DG, pos1)
        nx.draw_networkx_labels(DG, pos1)
        edge_labels1 = nx.get_edge_attributes(DG, 'weight')
        # 权重映射
        norm = Normalize(vmin=min(edge_labels1.values()), vmax=max(edge_labels1.values()))
        widths = [max(0.1, norm(weight) * 3) for weight in edge_labels1.values()]
        nx.draw_networkx_edges(DG, pos1, width=widths, edge_color='k', alpha=0.5)
        nx.draw_networkx_edge_labels(DG, pos1, label_pos=0.66, edge_labels=edge_labels1, rotate=True)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.text(0.05, 0.95, info, fontdict={'fontsize': 8}, transform=plt.gcf().transFigure)
        if not info.startswith('_'):  # Avoid to save so many repeated pics.
            log_name = info.split('Log Name:')[-1].replace('.log', '')
            plt.savefig((config.LOG_DIR + os.sep + log_name + os.sep + log_name + '_%s.svg' % str(dt.datetime.now())), format='svg')
        plt.show()













if __name__ == '__main__':
    import random as rd

    indiv = Business.remote(996, 1, {'cash': 1000000})
    print(indiv.observation_space.remote())

    # for i in range(10):
    #     indiv.step.remote(1)


