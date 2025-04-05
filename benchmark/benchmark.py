from collections import deque
from time import perf_counter
import platform

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT
from rtgym import RealTimeGymInterface
import gymnasium
from gymnasium.spaces import Box, Tuple


class BenchmarkInterface(RealTimeGymInterface):

    def __init__(self):
        self.control_timestamps = deque()
        self.obs_timestamps = deque()

    def get_observation_space(self):
        return Tuple((Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), ))

    def get_action_space(self):
        return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def get_default_action(self):
        return np.array([0.0], dtype=np.float32)

    def send_control(self, control):
        self.control_timestamps.append(perf_counter())

    def reset(self, seed=None, options=None):
        return [np.array([0.0], dtype=np.float32)], {}

    def get_obs_rew_terminated_info(self):
        self.obs_timestamps.append(perf_counter())
        return [np.array([0.0], dtype=np.float32)], 0.0, False, {}

    def wait(self):
        pass

    def compute_results(self):
        control_res = []
        obs_res = []

        for i, ts in enumerate(self.control_timestamps):
            if i == 0:
                t_last = ts
            else:
                t = ts - t_last
                t_last = ts
                control_res.append(t)

        for i, ts in enumerate(self.obs_timestamps):
            if i == 0:
                t_last = ts
            else:
                t = ts - t_last
                t_last = ts
                obs_res.append(t)

        return control_res, obs_res


target = 0.02
my_config = DEFAULT_CONFIG_DICT.copy()
my_config['interface'] = BenchmarkInterface
my_config['time_step_duration'] = target
my_config['start_obs_capture'] = target

env = gymnasium.make("real-time-gym-v1", config=my_config)

_, _ = env.reset()

act = np.array([0.0], dtype=np.float32)
for _ in range(1000):
    _, _, _, _, _ = env.step(act)

env.unwrapped.wait()

c, o = env.unwrapped.interface.compute_results()

print(np.mean(c))
print(np.mean(o))

df = pd.DataFrame(data={'control': np.array(c), 'capture': np.array(o)})

sns.set_style('whitegrid')
sns.displot(data=df, kde=True, bins=100).set(title=f'target: {target}, system:{platform.system()}')
plt.show()
