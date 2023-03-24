import unittest
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT
import time
import gymnasium
import numpy as np


class DummyInterface(RealTimeGymInterface):
    def __init__(self):
        self.control_time = None
        self.control = None

    def send_control(self, control):
        self.control_time = time.time()
        self.control = control

    def reset(self, seed=None, options=None):
        now = time.time()
        self.control_time = now
        self.control = [0.0]
        obs = [np.array([now], dtype=np.float64),
               np.array(self.control, dtype=np.float64),
               np.array([self.control_time], dtype=np.float64)]
        return obs, {}

    def get_obs_rew_terminated_info(self):
        obs = [np.array([time.time()], dtype=np.float64),
               np.array(self.control, dtype=np.float64),
               np.array([self.control_time], dtype=np.float64)]
        terminated = (self.control >= 9).item()
        return obs, 0.0, terminated, {}

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
config["start_obs_capture"] = 0.1
config["act_buf_len"] = 1
config["wait_on_done"] = False
config["reset_act_buf"] = False


class TestEnv(unittest.TestCase):
    def test_timing(self):
        epsilon = 0.02
        env = gymnasium.make("real-time-gym-v1", config=config)

        obs1, info = env.reset()
        elapsed_since_obs1_capture = time.time() - obs1[0]
        self.assertGreater(epsilon, elapsed_since_obs1_capture)

        # default action (buffer):
        self.assertEqual(obs1[3], -1)

        # arbitrary value:
        self.assertEqual(obs1[1], np.array([0.]))

        # elapsed between now and control time:
        now = time.time()
        self.assertGreater(0.1 + epsilon, now - obs1[0])

        act = np.array([0.0], dtype=np.float64)
        obs2, _, _, _, _ = env.step(act)
        self.assertEqual(obs2[3], act)
        self.assertEqual(obs2[1], -1.0)

        # elapsed between beginning of new timestep and previous obs capture:
        self.assertGreater(0.1 + epsilon, obs2[2] - obs1[0])

        # elapsed between new obs capture and previous obs capture:
        self.assertGreater(0.1 + epsilon, obs2[0] - obs1[0])

        for i in range(10):
            obs1 = obs2
            act = np.array([float(i + 1)])
            obs2, _, terminated, _, _ = env.step(act)
            now = time.time()
            self.assertEqual(obs2[3], act)
            self.assertEqual(obs2[1], act - 1.0)

            # elapsed between now and start of last timestep:
            self.assertGreater(0.1 + epsilon, now - obs2[2])

            # elapsed between new obs capture and previous obs capture:
            self.assertGreater(obs2[0] - obs1[0], 0.1 - epsilon)
            self.assertGreater(0.1 + epsilon, obs2[0] - obs1[0])

            # terminated signal:
            if i >= 9:
                self.assertTrue(terminated)

        # test reset:
        obs1, info = env.reset()

        # default action (buffer):
        self.assertEqual(obs1[3], -1)

        act = np.array([float(22)])
        obs1, _, _, _, _ = env.step(act)

        # new action (buffer):
        self.assertEqual(obs1[3], 22)


if __name__ == '__main__':
    unittest.main()
