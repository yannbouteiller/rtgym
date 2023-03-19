import unittest
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT
import time
import gymnasium
import numpy as np


class DummyInterface(RealTimeGymInterface):
    def __init__(self):
        self.control_time = 0
        self.control = [0.0]

    def send_control(self, control):
        self.control_time = time.time()
        self.control = control

    def reset(self, seed=None, options=None):
        obs = [np.array([time.time()], dtype=np.float64),
               np.array(self.control, dtype=np.float64),
               np.array([self.control_time], dtype=np.float64)]
        return obs, {}

    def get_obs_rew_terminated_info(self):
        obs = [np.array([time.time()], dtype=np.float64),
               np.array(self.control, dtype=np.float64),
               np.array([self.control_time], dtype=np.float64)]
        return obs, 0.0, False, {}

    def get_observation_space(self):
        ob = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float64)
        co = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float64)
        ct = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float64)
        return gymnasium.spaces.Tuple((ob, co, ct))

    def get_action_space(self):
        return gymnasium.spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), dtype=np.float64)

    def get_default_action(self):
        return np.array([-1.0], dtype=np.float64)


config = DEFAULT_CONFIG_DICT
config["interface"] = DummyInterface
config["time_step_duration"] = 0.1
config["start_obs_capture"] = 0.08
config["act_buf_len"] = 1


class TestEnv(unittest.TestCase):
    def test_timing(self):
        epsilon = 0.01
        env = gymnasium.make("real-time-gym-v1", config=config)
        obs1, info = env.reset()
        elapsed_since_obs1_capture = time.time() - obs1[0]
        self.assertGreater(epsilon, elapsed_since_obs1_capture)
        self.assertGreater(elapsed_since_obs1_capture, - epsilon)
        self.assertEqual(obs1[3], -1)
        self.assertEqual(obs1[1], np.array([0.]))
        self.assertEqual(obs1[2], np.array([0.]))
        act = np.array([0.0], dtype=np.float64)
        obs2, _, _, _, _ = env.step(act)
        self.assertEqual(obs2[3], act)
        self.assertEqual(obs2[1], -1.0)
        self.assertGreater(obs2[2] - obs1[0], - epsilon)
        self.assertGreater(epsilon, obs2[2] - obs1[0])
        self.assertGreater(obs2[0] - obs1[0], 0.08 - epsilon)
        self.assertGreater(0.08 + epsilon, obs2[0] - obs1[0])
        for i in range(3):
            obs1 = obs2
            act = np.array([float(i + 1)])
            obs2, _, _, _, _ = env.step(act)
            self.assertEqual(obs2[3], act)
            self.assertEqual(obs2[1], act - 1.0)
            self.assertGreater(time.time() - obs2[2], 0.1 - epsilon)
            self.assertGreater(0.1 + epsilon, time.time() - obs2[2])
            self.assertGreater(obs2[0] - obs1[0], 0.1 - epsilon)
            self.assertGreater(0.1 + epsilon, obs2[0] - obs1[0])


if __name__ == '__main__':
    unittest.main()
