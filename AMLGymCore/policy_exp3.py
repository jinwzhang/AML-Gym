import gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from AMLGymCore.businesses import business_v5 as business # environment
from stable_baselines3.common.callbacks import BaseCallback
import itertools

# 定义原始列表

# 使用 itertools.product 生成所有可能的组合

class TensorboardCallback(BaseCallback):
    def __int__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("record", self.training_env.get_attr("reward")[0])
        self.logger.record("episode_length", self.training_env.get_attr(("episode_length")[0]))
        return True


GROUP_matrix = [[7]]
AML_matrix = [(False, False), (True, False), (True, True)]

combined_matrix = list(itertools.product(GROUP_matrix, AML_matrix))
algorithms =[(DQN, 'dqn'), (PPO, "ppo"), (A2C, 'a2c')]


for algo, name in algorithms:

    for item in combined_matrix:
        print(name, item, ' start')
        group1, aml = item
        aml_check1, aml_random1 = aml
        env = business.Business(train_memo=name + str(item), group=group1, aml_policy=aml_check1, aml_random_check=aml_random1)

        model = algo('MlpPolicy', env, verbose=1)
        policy = model.policy
        print(policy)
        model.learn(total_timesteps=1000000)
        model.save(name + '_model')
        env.close()
        print(name, 'Done')
        loaded_model = algo.load(name + '_model')

        # mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=100)
        # print('mean_reward: ',mean_reward, ' std_reward:',std_reward )
        # with open('/data/log_AML/202403/group1/%s.log' % name, 'w') as f:
        #     f.write(str(mean_reward))
