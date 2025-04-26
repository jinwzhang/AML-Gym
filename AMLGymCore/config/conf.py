from os import sep

INGENSTION = {'cash': 1000000}
TRANSACTION_VALUE = 5000   # Important - this is the config amount of every transaction.
MAX_STEP = 2000
GROUP = [4]   # list or value - represent which industries the agent could enter
AML_POLICY = False
AML_RAMDOM_CHECK = False

BASE_DIR = r'/data/rl/AMLGym/AMLGymCore' + sep
ENV_CONFIG_EXCEL = BASE_DIR + sep + 'config' + sep + 'business_config_template_v2.xlsx'
LOG_DIR = '/data/log_AML/' + '202404/all_group'

