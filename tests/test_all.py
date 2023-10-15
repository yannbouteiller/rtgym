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
        self.control = [-2.0]
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


class TestEnv(unittest.TestCase):

    def timing(self, env_str):
        epsilon = 0.02
        act_buf_len = 3
        time_step_duration = 0.1
        start_obs_capture = 0.08

        print("--- new environment ---")

        config = DEFAULT_CONFIG_DICT
        config["interface"] = DummyInterface
        config["time_step_duration"] = time_step_duration
        config["start_obs_capture"] = start_obs_capture
        config["act_buf_len"] = act_buf_len
        config["wait_on_done"] = False
        config["reset_act_buf"] = False
        config["ep_max_length"] = 10
        config["last_act_on_reset"] = True

        env = gymnasium.make(env_str, config=config)

        # first reset, the default action (-1) will be sent:
        obs1, info = env.reset()

        # now, action -1 is on its way

        # what is the difference between now and the moment reset was called?
        now = time.time()
        elapsed_since_reset = now - obs1[0]
        print(f"Call to reset took {elapsed_since_reset} seconds")
        self.assertGreater(epsilon, elapsed_since_reset)

        # the actions in the buffer should all be the default (-1):
        for j in range(act_buf_len):
            print(f"The action buffer is {obs1[3 + j]} at index {j}")
            self.assertEqual(obs1[3 + j], -1)

        # the first control is -2 at reset:
        print(f"The control is {obs1[1]}")
        self.assertEqual(obs1[1], np.array([-2.]))

        # now let us step the environment
        a = 1
        print(f"--- Step {a} ---")
        act = np.array([float(a)], dtype=np.float64)
        obs2, _, terminated, truncated, _ = env.step(act)
        now = time.time()
        print(f"terminated: {terminated}, truncated:{truncated}")

        # let us look at the action buffer:
        for j in range(act_buf_len):
            print(f"The action buffer is {obs2[3 + j]} at index {j}")
            if j < act_buf_len - 1:
                self.assertEqual(obs2[3 + j], -1)
            else:
                self.assertEqual(obs2[3 + j], a)

        # Now, we look at the time elapsed between the observation retrieval of step and reset:
        elapsed = obs2[0] - obs1[0]
        print(f"The two last obs are spaced by {elapsed} seconds")
        self.assertGreater(time_step_duration + epsilon, elapsed)
        self.assertGreater(elapsed, start_obs_capture - epsilon)

        # the control applied when obs2 was captured should be the default -1
        print(f"The action applied when obs was captured was {obs2[1]}")
        self.assertEqual(obs2[1], np.array([-1.]))

        # the sending timestamp of the control should be the beginning of the last time-step:
        elapsed = now - obs2[2]
        print(f"This action was sent {elapsed} seconds ago")
        self.assertGreater(time_step_duration + epsilon, elapsed)
        self.assertGreater(elapsed, time_step_duration - epsilon)

        for i in range(9):

            obs1 = obs2
            a += 1

            print(f"--- Step {a} ---")
            act = np.array([float(a)], dtype=np.float64)
            obs2, _, terminated, truncated, _ = env.step(act)
            now = time.time()
            print(f"terminated: {terminated}, truncated:{truncated}")

            # let us look at the action buffer:
            for j in range(act_buf_len):
                print(f"The action buffer is {obs2[3 + j]} at index {j}")
            self.assertEqual(obs2[-1], a)

            # Now, we look at the time elapsed between the two observations:
            elapsed = obs2[0] - obs1[0]
            print(f"The two last obs are spaced by {elapsed} seconds")
            self.assertGreater(time_step_duration + epsilon, elapsed)
            self.assertGreater(elapsed, time_step_duration - epsilon)

            # the control applied when obs2 was captured should be the previous a
            print(f"The action applied when obs was captured was {obs2[1]}")
            self.assertEqual(obs2[1], np.array([float(a - 1)]))

            # the sending timestamp of the control should be the beginning of the last time-step:
            elapsed = now - obs2[2]
            print(f"This action was sent {elapsed} seconds ago")
            self.assertGreater(time_step_duration + epsilon, elapsed)
            self.assertGreater(elapsed, time_step_duration - epsilon)

            # end of episode:
            if i == 8:
                # the terminated signal should override the truncated signal:
                self.assertTrue(terminated)
                self.assertFalse(truncated)

        # let us test the real-time reset mechanism:
        print("--- reset ---")
        obs1, info = env.reset()
        now = time.time()

        # this call to reset should be near-instantaneous:
        elapsed_since_reset = now - obs1[0]
        print(f"Call to reset took {elapsed_since_reset} seconds")
        self.assertGreater(epsilon, elapsed_since_reset)

        # let us look at the action buffer:
        for j in range(act_buf_len):
            print(f"The action buffer is {obs1[3 + j]} at index {j}")
        # since we use "last_act_on_reset":True, the buffer should end with act:
        self.assertEqual(obs1[-1], act)

        # Now let us step the environment:

        a = 0
        act = np.array([float(a)], dtype=np.float64)
        obs1, _, terminated, truncated, _ = env.step(act)

        # because we sent the previous act (10) on reset(), terminated should now be True:
        print(f"terminated: {terminated}, truncated:{truncated}")
        self.assertEqual(terminated, True)
        self.assertEqual(truncated, False)

        # Let us retry:

        print("--- reset ---")
        obs1, info = env.reset()

        for i in range(10):
            print(f"--- step {i + 1} ---")
            obs1, _, terminated, truncated, _ = env.step(act)
            print(f"terminated: {terminated}, truncated:{truncated}")
            if i < 9:
                # reset() sent 0, so we should be good:
                self.assertEqual(terminated, False)
                self.assertEqual(truncated, False)
            else:
                # the episode should now be truncated:
                self.assertEqual(terminated, False)
                self.assertEqual(truncated, True)

        # Now the episode is truncated, we should not be able to call step again:

        try:
            obs1, _, terminated, truncated, _ = env.step(act)
            assert False, "step did not raise a RuntimeError"
        except RuntimeError:
            print("step cannot be called here.")

        # Now let us test the default reset behavior:

        print("--- new environment ---")

        config = DEFAULT_CONFIG_DICT
        config["interface"] = DummyInterface
        config["time_step_duration"] = time_step_duration
        config["start_obs_capture"] = start_obs_capture
        config["act_buf_len"] = act_buf_len
        config["wait_on_done"] = True
        config["reset_act_buf"] = True
        config["ep_max_length"] = 10
        config["last_act_on_reset"] = False

        env = gymnasium.make("real-time-gym-v1", config=config)

        # reset, the default action (-1) will be sent:
        obs1, info = env.reset()

        # the actions in the buffer should all be the default (-1):
        for j in range(act_buf_len):
            print(f"The action buffer is {obs1[3 + j]} at index {j}")
            self.assertEqual(obs1[3 + j], -1)

        # the first control is -2 at reset:
        print(f"The control is {obs1[1]}")
        self.assertEqual(obs1[1], np.array([-2.]))

        # let us step:
        print("--- step ---")
        obs1, _, terminated, truncated, _ = env.step(act)

        # let us look at the action buffer:
        for j in range(act_buf_len):
            print(f"The action buffer is {obs1[3 + j]} at index {j}")
        # the buffer should end with act:
        self.assertEqual(obs1[-1], act)

        # let us step again:
        print("--- step ---")
        obs1, _, terminated, truncated, _ = env.step(act)

        # let us look at the action buffer:
        for j in range(act_buf_len):
            print(f"The action buffer is {obs1[3 + j]} at index {j}")
        # the buffer should still end with act:
        self.assertEqual(obs1[-1], act)

        # not let us reset again:
        print("--- reset ---")
        obs1, info = env.reset()

        # the actions in the buffer should now all be the default (-1):
        for j in range(act_buf_len):
            print(f"The action buffer is {obs1[3 + j]} at index {j}")
            self.assertEqual(obs1[3 + j], -1)

        # and the first control is -2 at reset:
        print(f"The control is {obs1[1]}")
        self.assertEqual(obs1[1], np.array([-2.]))

        # for good measure, let us step one last time:

        print("--- step ---")
        obs1, _, terminated, truncated, _ = env.step(act)

        # the last actions in the buffer should be act:
        for j in range(act_buf_len):
            print(f"The action buffer is {obs1[3 + j]} at index {j}")
        self.assertEqual(obs1[-1], -act)

        # the applied control should be -1:
        print(f"The action applied when obs was captured was {obs1[1]}")
        self.assertEqual(obs1[1], np.array([-1.]))

        # now let us test changing the configuration on-the-fly

        # reset:
        print("--- reset ---")
        obs1, info = env.reset()

        # now let us step the environment
        a = 1
        print(f"--- Step {a} ---")
        act = np.array([float(a)], dtype=np.float64)
        obs2, _, terminated, truncated, _ = env.step(act)
        print(f"terminated: {terminated}, truncated:{truncated}")

        for i in range(3):

            obs1 = obs2
            a += 1

            print(f"--- Step {a} ---")
            act = np.array([float(a)], dtype=np.float64)
            obs2, _, terminated, truncated, _ = env.step(act)
            now = time.time()
            print(f"terminated: {terminated}, truncated:{truncated}")

            # let us look at the action buffer:
            for j in range(act_buf_len):
                print(f"The action buffer is {obs2[3 + j]} at index {j}")
            self.assertEqual(obs2[-1], a)

            # Now, we look at the time elapsed between the two observations:
            elapsed = obs2[0] - obs1[0]
            print(f"The two last obs are spaced by {elapsed} seconds")
            self.assertGreater(time_step_duration + epsilon, elapsed)
            self.assertGreater(elapsed, time_step_duration - epsilon)

            # the control applied when obs2 was captured should be the previous a
            print(f"The action applied when obs was captured was {obs2[1]}")
            self.assertEqual(obs2[1], np.array([float(a - 1)]))

            # the sending timestamp of the control should be the beginning of the last time-step:
            elapsed = now - obs2[2]
            print(f"This action was sent {elapsed} seconds ago")
            self.assertGreater(time_step_duration + epsilon, elapsed)
            self.assertGreater(elapsed, time_step_duration - epsilon)

        new_time_step_duration = 0.2
        new_ep_max_length = 8
        print(f"changing time step duration to {new_time_step_duration} and episode length to {new_ep_max_length}")
        # let us change the parameters (this waits for the end of the ongoing time step):
        env.unwrapped.set_time_step_duration(time_step_duration=new_time_step_duration)
        env.unwrapped.set_start_obs_capture(start_obs_capture=new_time_step_duration)
        env.unwrapped.set_ep_max_length(ep_max_length=new_ep_max_length)

        # the next time step will still be of the old duration:

        obs1 = obs2
        a += 1

        print(f"--- Step {a} ---")
        act = np.array([float(a)], dtype=np.float64)
        obs2, _, terminated, truncated, _ = env.step(act)
        now = time.time()
        print(f"terminated: {terminated}, truncated:{truncated}")

        # let us look at the action buffer:
        for j in range(act_buf_len):
            print(f"The action buffer is {obs2[3 + j]} at index {j}")
        self.assertEqual(obs2[-1], a)

        # Now, we look at the time elapsed between the two observations:
        elapsed = obs2[0] - obs1[0]
        print(f"The two last obs are spaced by {elapsed} seconds")
        self.assertGreater(time_step_duration + epsilon, elapsed)
        self.assertGreater(elapsed, time_step_duration - epsilon)

        # the control applied when obs2 was captured should be the previous a
        print(f"The action applied when obs was captured was {obs2[1]}")
        self.assertEqual(obs2[1], np.array([float(a - 1)]))

        # the sending timestamp of the control should be the beginning of the last time-step:
        elapsed = now - obs2[2]
        print(f"This action was sent {elapsed} seconds ago")
        self.assertGreater(time_step_duration + epsilon, elapsed)
        self.assertGreater(elapsed, time_step_duration - epsilon)

        # but the following time steps will be of the new duration, and episode will end on time step 8

        for i in range(5):

            obs1 = obs2
            a += 1

            print(f"--- Step {a} ---")
            act = np.array([float(a)], dtype=np.float64)
            obs2, _, terminated, truncated, _ = env.step(act)
            now = time.time()
            print(f"terminated: {terminated}, truncated:{truncated}")

            # let us look at the action buffer:
            for j in range(act_buf_len):
                print(f"The action buffer is {obs2[3 + j]} at index {j}")
            self.assertEqual(obs2[-1], a)

            # Now, we look at the time elapsed between the two observations:
            elapsed = obs2[0] - obs1[0]
            print(f"The two last obs are spaced by {elapsed} seconds")
            if i > 0:  # there is some jitter on first iteration
                self.assertGreater(new_time_step_duration + epsilon, elapsed)
                self.assertGreater(elapsed, new_time_step_duration - epsilon)

            # the control applied when obs2 was captured should be the previous a
            print(f"The action applied when obs was captured was {obs2[1]}")
            self.assertEqual(obs2[1], np.array([float(a - 1)]))

            # the sending timestamp of the control should be the beginning of the last time-step:
            elapsed = now - obs2[2]
            print(f"This action was sent {elapsed} seconds ago")
            if i > 0:  # there is some jitter on first iteration
                self.assertGreater(new_time_step_duration + epsilon, elapsed)
                self.assertGreater(elapsed, new_time_step_duration - epsilon)

            # end of episode:
            if a == 8:
                # the terminated signal should override the truncated signal:
                self.assertFalse(terminated)
                self.assertTrue(truncated)
                break

    def test_timing_v1(self):
        self.timing(env_str="real-time-gym-v1")

    def test_timing_ts_v1(self):
        self.timing(env_str="real-time-gym-ts-v1")


if __name__ == '__main__':
    unittest.main()
