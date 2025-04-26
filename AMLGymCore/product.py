import pandas as pd
from AMLGymCore.utils.functions import *

trade_config_df = pd.read_excel(conf.ENV_CONFIG_EXCEL, sheet_name='trade')
trade_config_df = trade_config_df.sample(frac=1)  # 20230901 Add config rows sequence shuffle
trade_config_df.dropna(axis=0, subset='exchangeable_target', inplace=True)
trade_config_df.dropna(axis=0, subset='product_name', inplace=True)
trade_config_df.drop_duplicates(subset='trade_id', inplace=True)
trade_config_df.fillna(method='ffill')
trade_config_df.loc[:, 'trade_id'] = trade_config_df['trade_id'].apply(lambda s: str(s).lower().strip())
trade_config_df.loc[:, 'product_name'] = trade_config_df['product_name'].apply(lambda s: str(s).lower().strip())
# To load the json in config table into dict.
trade_config_df['_exchangeable_target'] = trade_config_df['exchangeable_target'].apply(multi_target_loader)
trade_config_df['_product_name'] = trade_config_df['product_name'].apply(multi_target_loader)


class TradeType:
    def __init__(self, trade_id):
        self.trade_id = trade_id

        self.attributes = trade_config_df[trade_config_df['trade_id'] == self.trade_id]
        try:
            self.attributes = self.attributes.to_dict('records')[0]
        except Exception:
            raise ValueError('Trade_ID %s config error' % self.trade_id)
        self.product_name = self.attributes.get('_product_name')
        self.exchangeable_target = self.attributes.get('_exchangeable_target')
        self.one_time_cost = self.attributes.get('one_time_cost', 0)
        self.memo = self.attributes.get('memo')
        self.group = self.attributes.get('group')
        self.max_volume = self.attributes.get('max_volume', np.inf)
        self.max_volume = np.inf if self.max_volume == 0 else self.max_volume
        # transaction_level

        self.paid_value = self.product_name
        self.return_value = 0
        self.reportable_profit = 0
        self.ret_dict = {}
        # risk_level
        # self.normal_amount = self.attributes.get('_normal_amount', np.inf)
        # self.normal_amount += self.one_time_cost
        self.abnormal_report = self.attributes.get('abnormal_report', 'N')
        self.do_abnormal_report = 'N'
        self.transaction_risk_function = self.attributes.get('transaction_risk_function', 'N')
        self.common_sense_abnormal_factor = 0


    def trasaction(self, amount):
        # In product level transactions
        if self.max_volume <= 0:   # Can't exceed product's max volume setting.
            amount = 0
        else:
            amount = min(amount, self.max_volume)

        self.max_volume -= amount
        self.paid_value = amount
        value_after_one_time_cost = max(0, amount - self.one_time_cost)
        self.ret_dict = {k: v * value_after_one_time_cost for k, v in self.exchangeable_target.items()}

        if self.ret_dict.get('cost'):
            self.ret_dict['cost'] += self.one_time_cost
        else:
            self.ret_dict['cost'] = self.one_time_cost
        return self.paid_value, self.ret_dict

    # def product_risk_reveal(self):
    #     self.common_sense_abnormal_factor = self.paid_value / self.normal_amount
    #     if self.abnormal_report == 'Y' and self.common_sense_abnormal_factor >= 1:
    #         if rd.random() * self.common_sense_abnormal_factor > 0.3:
    #             self.do_abnormal_report = 'Y'
    #     else:
    #         if rd.random() * self.common_sense_abnormal_factor > 0.9:
    #             self.do_abnormal_report = 'Y'
    #     return self.do_abnormal_report, self.common_sense_abnormal_factor

    # def transaction_check(self, transaction_his):
    #     transaction_total = transaction_his.get(self.trade_id, 0) * 9000
    #     banned = True if self.max_volume <= transaction_total else False
    #     observable = True if self.normal_amount <= transaction_total and self.abnormal_report == 'Y' else False
    #     return banned, observable








