"""Real-Time Gym environment core.

The final environment instantiated by gymnasium.make is RealTimeEnv.
The developer defines a custom implementation of the RealTimeGymInterface abstract class.
Then, they create a config dictionary by copying DEFAULT_CONFIG_DICT.
In this config, they replace the 'interface' entry with their custom RealTimeGymInterface.
Their custom RealTimeGymInterface may implement __init__(*args, **kwargs).
The args and kwargs are passed through the 'interface_args' and 'interface_kwargs' entries of the config.
Other entries define Real-Time Gym parameters, such as the nominal duration of the elastic time-step.
"""


from gymnasium import Env
import gymnasium.spaces as spaces
import time
from collections import deque
from threading import Thread, Lock
import warnings
import numpy as np


# General Interface class ==============================================================================================
# All user-defined interfaces should be subclasses of RealTimeGymInterface

class RealTimeGymInterface:
    """Main developer interface.

    Implement this class for your application.
    Then copy the DEFAULT_CONFIG_DICT and replace the 'interface' entry with your custom RealTimeGymInterface.
    """
    def send_control(self, control):
        """Sends control to the device.

        Non-blocking function.
        Applies the action given by the RL policy.
        If control is None, should do nothing.
        e.g.
        if control is not None:
            ...

        Args:
            control: np.array of the dimension of the action-space (possibly None: do nothing)
        """
        # if control is not None:
        #     ...

        raise NotImplementedError

    def reset(self, seed=None, options=None):
        """Resets the episode.

        Args:
            seed (optional): Gymnasium seed
            options (optional): Gymnasium options

        Returns:
            obs: must be a list (corresponding to the tuple from get_observation_space)
            info: must be a dictionary

        Note: Do NOT put the action buffer in the returned obs (automated).
        """
        # return obs, info

        raise NotImplementedError

    def wait(self):
        """The environment 'waits' for the next reset().

        Implement this when you want to artificially 'pause' your environment after an episode.

        CAUTION: If you implement this, make sure 'reset_act_buf' is True in your rtgym configuration dictionary.
        """
        pass

    def get_obs_rew_terminated_info(self):
        """Returns observation, reward, terminated and info from the device.

        Note:

        Returns:
            obs: list (corresponding to the tuple from get_observation_space)
            rew: scalar
            terminated: bool
            info: dict

        Note: Do NOT put the action buffer in obs (automated).
        """
        # return obs, rew, terminated, info

        raise NotImplementedError

    def get_observation_space(self):
        """Returns the observation space.

        Returns:
            observation_space: gymnasium.spaces.Tuple

        Note: Do NOT put the action buffer here (automated).
        """
        # return spaces.Tuple(...)

        raise NotImplementedError

    def get_action_space(self):
        """Returns the action space.

        Returns:
            action_space: gymnasium.spaces.Box
        """
        # return spaces.Box(...)

        raise NotImplementedError

    def get_default_action(self):
        """Initial action at episode start, and in the buffer.

        Returns:
            default_action: numpy array of the dimension of the action space
        """
        # return np.array([...], dtype='float32')

        raise NotImplementedError

    def render(self):
        """Renders the environment (optional).

        Implement this if you want to use the render() method of the gymnasium environment.
        """
        pass


# General purpose environment: =========================================================================================

DEFAULT_CONFIG_DICT = {
    "interface": RealTimeGymInterface,  # replace this by your custom interface class
    "interface_args": (),  # arguments of your interface
    "interface_kwargs": {},  # key word arguments of your interface
    "time_step_duration": 0.05,  # nominal duration of your time-step
    "start_obs_capture": 0.05,  # observation retrieval will start this amount of time after the time-step begins
    # start_obs_capture should be the same as "time_step_duration" unless observation capture is non-instantaneous and
    # smaller than one time-step, and you want to capture it directly in your interface for convenience. Otherwise,
    # you need to perform observation capture in a parallel process and simply retrieve the last available observation
    # in the get_obs_rew_terminated_info() and reset() methods of your interface
    "time_step_timeout_factor": 1.0,  # maximum elasticity in (fraction or number of) time-steps
    "ep_max_length": np.inf,  # maximum episode length
    "real_time": True,  # True unless you want to revert to the usual turn-based RL setting (not tested yet)
    "async_threading": True,  # True unless you want to revert to the usual turn-based RL setting (not tested yet)
    "act_in_obs": True,  # When True, the action buffer will be appended to observations
    "act_buf_len": 1,  # Length of the action buffer (should be max total delay + max observation capture duration, in time-steps)
    "reset_act_buf": True,  # When True, the action buffer will be filled with default actions at reset
    "benchmark": False,  # When True, a simple benchmark will be run to estimate useful timing metrics
    "benchmark_polyak": 0.1,  # Polyak averaging factor for the benchmarks (0.0 < x <= 1); smaller is slower, bigger is noisier
    "wait_on_done": False,  # Whether the wait() method should be called when either terminated or truncated is True
    "last_act_on_reset": False,  # When False, reset() sends the default action; when False, it sends the last action
}
"""Default configuration dictionary of Real-Time Gym.

Copy this dictionary and pass it as argument to gymnasium.make.

    Typical usage example:

    my_config = DEFAULT_CONFIG_DICT
    my_config["interface"] = MyRealTimeInterface
    env = gymnasium.make("real-time-gym-v1", config=my_config)
"""


class Benchmark:
    """Benchmarks of the durations of main operations.

    A running average of the mean and average deviation are provided for each duration.
    The results are returned as a dictionary of (average, average_deviation) entries.
    """
    def __init__(self, run_avg_factor=0.1):
        self.run_avg_factor = run_avg_factor
        self.__b_lock = Lock()

        # times:
        self.__start_step_time = None
        self.__end_step_time = None
        self.__start_retrieve_obs_time = None
        self.__end_retrieve_obs_time = None
        self.__start_time_step_time = None
        self.__end_send_control_time = None

        # averages:
        self.time_step_duration = None
        self.step_duration = None
        self.join_duration = None
        self.inference_duration = None
        self.send_control_duration = None
        self.retrieve_obs_duration = None

        # average deviations:
        self.time_step_duration_dev = 0.0
        self.step_duration_dev = 0.0
        self.join_duration_dev = 0.0
        self.inference_duration_dev = 0.0
        self.send_control_duration_dev = 0.0
        self.retrieve_obs_duration_dev = 0.0

    def get_benchmark_dict(self):
        """Each key contains a tuple (avg, avg_dev)
        """
        self.__b_lock.acquire()
        res = {
            "time_step_duration": (self.time_step_duration, self.time_step_duration_dev),
            "step_duration": (self.step_duration, self.step_duration_dev),
            "join_duration": (self.join_duration, self.join_duration_dev),
            "inference_duration": (self.inference_duration, self.inference_duration_dev),
            "send_control_duration": (self.send_control_duration, self.send_control_duration_dev),
            "retrieve_obs_duration": (self.retrieve_obs_duration, self.retrieve_obs_duration_dev),
        }
        self.__b_lock.release()
        return res

    def running_average(self, new_val, old_avg, old_avg_dev):
        """Running average.

        old_avg can be None
        new_avg_dev is the average deviation (not std)

        Args:
            new_val, old_avg, old_avg_dev

        Returns:
            new_avg, new_avg_dev
        """
        if old_avg is not None:
            delta = new_val - old_avg
            new_avg = old_avg + self.run_avg_factor * delta
            new_avg_dev = old_avg_dev + self.run_avg_factor * (abs(delta) - old_avg_dev)
            return new_avg, new_avg_dev
        else:
            return new_val, 0.0

    def start_step_time(self):
        """before join().
        """
        self.__b_lock.acquire()
        now = time.time()
        if self.__end_step_time is not None:
            self.inference_duration, self.inference_duration_dev = self.running_average(new_val=now - self.__end_step_time, old_avg=self.inference_duration, old_avg_dev=self.inference_duration_dev)
        self.__start_step_time = now
        self.__b_lock.release()

    def end_step_time(self):
        """before return.
        """
        self.__b_lock.acquire()
        now = time.time()
        if self.__start_step_time is not None:
            self.step_duration, self.step_duration_dev = self.running_average(new_val=now - self.__start_step_time, old_avg=self.step_duration, old_avg_dev=self.step_duration_dev)
        self.__end_step_time = now
        self.__b_lock.release()

    def start_time_step_time(self):
        """before run_time_step.
        """
        self.__b_lock.acquire()
        now = time.time()
        if self.__start_time_step_time is not None:
            self.time_step_duration, self.time_step_duration_dev = self.running_average(new_val=now - self.__start_time_step_time, old_avg=self.time_step_duration, old_avg_dev=self.time_step_duration_dev)
        if self.__start_step_time is not None:
            self.join_duration, self.join_duration_dev = self.running_average(new_val=now - self.__start_step_time, old_avg=self.join_duration, old_avg_dev=self.join_duration_dev)
        self.__start_time_step_time = now
        self.__b_lock.release()

    def start_retrieve_obs_time(self):
        self.__b_lock.acquire()
        now = time.time()
        self.__start_retrieve_obs_time = now
        self.__b_lock.release()

    def end_retrieve_obs_time(self):
        self.__b_lock.acquire()
        now = time.time()
        if self.__start_retrieve_obs_time is not None:
            self.retrieve_obs_duration, self.retrieve_obs_duration_dev = self.running_average(new_val=now - self.__start_retrieve_obs_time, old_avg=self.retrieve_obs_duration, old_avg_dev=self.retrieve_obs_duration_dev)
        self.__end_retrieve_obs_time = now
        self.__b_lock.release()

    def end_send_control_time(self):
        self.__b_lock.acquire()
        now = time.time()
        if self.__start_time_step_time is not None:
            self.send_control_duration, self.send_control_duration_dev = self.running_average(new_val=now - self.__start_time_step_time, old_avg=self.send_control_duration, old_avg_dev=self.send_control_duration_dev)
        self.__end_send_control_time = now
        self.__b_lock.release()


class RealTimeEnv(Env):
    def __init__(self, config: dict=DEFAULT_CONFIG_DICT):
        """Final class instantiated by gymnasium.make.

        Args:
            config: a custom implementation of DEFAULT_CONFIG_DICT
        """
        # interface:
        interface_cls = config["interface"]
        interface_args = config["interface_args"] if "interface_args" in config else ()
        interface_kwargs = config["interface_kwargs"] if "interface_kwargs" in config else {}
        self.interface = interface_cls(*interface_args, **interface_kwargs)
        self.is_waiting = False

        # config variables:
        self.wait_on_done = config["wait_on_done"] if "wait_on_done" in config else False
        self.last_act_on_reset = config["last_act_on_reset"] if "last_act_on_reset" in config else False
        self.act_prepro_func: callable = config["act_prepro_func"] if "act_prepro_func" in config else None
        self.obs_prepro_func = config["obs_prepro_func"] if "obs_prepro_func" in config else None
        self.ep_max_length = config["ep_max_length"]

        self.time_step_duration = config["time_step_duration"] if "time_step_duration" in config else 0.0
        self.time_step_timeout_factor = config["time_step_timeout_factor"] if "time_step_timeout_factor" in config else 1.0
        self.start_obs_capture = config["start_obs_capture"] if "start_obs_capture" in config else 1.0
        self.time_step_timeout = self.time_step_duration * self.time_step_timeout_factor  # time after which elastic time-stepping is dropped
        self.real_time = config["real_time"]
        self.async_threading = config["async_threading"] if "async_threading" in config else True
        self.__t_start = None  # beginning of the time-step
        self.__t_co = None  # time at which observation starts being captured during the time step
        self.__t_end = None  # end of the time-step
        if not self.real_time:
            self.async_threading = False
        if self.async_threading:
            self._at_thread = Thread(target=None, args=(), kwargs={}, daemon=True)
            self._at_thread.start()  # dummy start for later call to join()

        # observation capture:
        self.__o_lock = Lock()  # lock to retrieve observations asynchronously, acquire to access the following:
        self.__obs = None
        self.__rew = None
        self.__terminated = None
        self.__info = None
        self.__o_set_flag = False

        # environment benchmark:
        self.benchmark = config["benchmark"] if "benchmark" in config else False
        self.benchmark_polyak = config["benchmark_polyak"] if "benchmark_polyak" in config else 0.1
        self.bench = Benchmark(run_avg_factor=self.benchmark_polyak)

        self.act_in_obs = config["act_in_obs"] if "act_in_obs" in config else True
        self.act_buf_len = config["act_buf_len"] if "act_buf_len" in config else 1
        self.act_buf = deque(maxlen=self.act_buf_len)
        self.reset_act_buf = config["reset_act_buf"] if "reset_act_buf" in config else True
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.current_step = 0
        self.time_initialized = False
        self.running = False

        # state variables:
        self.default_action = self.interface.get_default_action()

        # gymnasium variables:
        self.seed = None
        self.options = None

        # action buffer initialization:
        self.init_action_buffer()

    def _update_timestamps(self):
        """This is called at the beginning of each time-step.

        If the previous time-step has timed out, the beginning of the time-step is set to now
        Otherwise, the beginning of the time-step is the beginning of the previous time-step + the time-step duration
        The observation starts being captured start_obs_capture_factor time-step after the beginning of the time-step
            observation capture can exceed the time-step, it is fine, but be cautious with timeouts
        It is recommended to draw a time diagram of your system
            action computation and observation capture can be performed in parallel
        """
        now = time.time()
        if now < self.__t_end + self.time_step_timeout:  # if either still in the previous time-step of within its allowed elasticity
            self.__t_start = self.__t_end  # the new time-step starts when the previous time-step is supposed to finish or to have finished
        else:  # if after the allowed elasticity
            if not self.is_waiting:
                warnings.warn(f"Time-step timed out. Elapsed since last time-step: {now - self.__t_end}")
            else:
                self.is_waiting = False
            self.__t_start = now  # the elasticity is broken and reset (this should happen only after 'pausing' the environment)
        self.__t_co = self.__t_start + self.start_obs_capture  # update time at which observation should be retrieved
        self.__t_end = self.__t_start + self.time_step_duration  # update time at which the new time-step should finish

    def _join_thread(self):
        """This is called at the beginning of every user-side API functions (step(), reset()...) for thread safety.

        This ensures that the previous time-step is completed when starting a new one
        """
        if self.async_threading:
            self._at_thread.join()

    def _run_time_step(self, *args, **kwargs):
        """This is called in step() to apply an action.

        Call this with the args and kwargs expected by self.__send_act_get_obs_and_wait()
        This in turn calls self.__send_act_get_obs_and_wait()
        In action-threading, self.__send_act_get_obs_and_wait() is called in a new Thread
        """
        if not self.async_threading:
            self.__send_act_get_obs_and_wait(*args, **kwargs)
        else:
            self._at_thread = Thread(target=self.__send_act_get_obs_and_wait, args=args, kwargs=kwargs, daemon=True)
            self._at_thread.start()

    def _initialize_time(self):
        """This is called at first reset()."""
        now = time.time()
        # fake a "previous" time step:
        self.__t_start = now - self.time_step_duration
        self.__t_co = self.__t_start + self.start_obs_capture
        self.__t_end = self.__t_start + self.time_step_duration
        self.time_initialized = True

    def _get_action_space(self):
        return self.interface.get_action_space()

    def _get_observation_space(self):
        t = self.interface.get_observation_space()
        if self.act_in_obs:
            t = spaces.Tuple((*t.spaces, *((self._get_action_space(),) * self.act_buf_len)))
        return t

    def __send_act_get_obs_and_wait(self, action):
        """Applies the control and launches observation capture at the right timestamp.

        Caution: only one such function must run in parallel (always join thread)
        """
        if self.benchmark:
            self.bench.start_time_step_time()
        act = self.act_prepro_func(action) if self.act_prepro_func else action
        self.interface.send_control(act)
        if self.benchmark:
            self.bench.end_send_control_time()
        self._update_timestamps()
        now = time.time()
        if now < self.__t_co:  # wait until it is time to capture observation
            time.sleep(self.__t_co - now)
        if self.benchmark:
            self.bench.start_retrieve_obs_time()
        self.__update_obs_rew_terminated_truncated()  # capture observation
        if self.benchmark:
            self.bench.end_retrieve_obs_time()
        now = time.time()
        if now < self.__t_end:  # wait until the end of the time-step
            time.sleep(self.__t_end - now)

    def __update_obs_rew_terminated_truncated(self):
        """Captures o, r, d asynchronously.

        Returns:
            observation of this step()
        """
        self.__o_lock.acquire()
        o, r, d, i = self.interface.get_obs_rew_terminated_info()
        elt = o
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        self.__obs, self.__rew, self.__terminated, self.__info = elt, r, d, i
        self.__o_set_flag = True
        self.__o_lock.release()

    def _retrieve_obs_rew_terminated_info(self):
        """Waits for new available o r d i and retrieves them.
        """
        c = True
        while c:
            self.__o_lock.acquire()
            if self.__o_set_flag:
                elt, r, d, i = self.__obs, self.__rew, self.__terminated, self.__info
                self.__o_set_flag = False
                c = False
            self.__o_lock.release()
        return elt, r, d, i

    def init_action_buffer(self):
        for _ in range(self.act_buf_len):
            self.act_buf.append(self.default_action)

    def reset(self, seed=None, options=None):
        """Resets the environment.

        The reset() function has a special role in the Real-Time setting.
        It applies an initial action, sine one action is applied at all time in real-time environments.
        This initial action is the default_action of your environment.
        You may or may not wish the rtgym elastic clock to keep running during reset transitions.

        If you wish rtgym to break real-time constraints after an episode, you can either set 'wait_on_done' to True in
        your config, or call the wait() method of your environment whenever terminated or truncated is True.
        Customize the wait() function if you wish your environment to do something before you call reset().

        If on the contrary you wish to keep the real-time constraints from one episode to the next, your reset()
        interface implementation should capture an observation almost instantaneously. This is because your previous
        action will keep being applied until your observation is captured.

        Args:
            seed (optional): seed passed to the reset() method of the rtgym interface
            options (optional): option dictionary passed to the reset() method of the rtgym interface

        Returns:
            obs: first observation of the trajectory, including real-time action buffer
            info: info dictionary
        """
        self._join_thread()
        self.running = True
        self.seed = seed
        self.options = options
        self.current_step = 0
        if self.reset_act_buf:
            # fill the action buffer with default actions:
            self.init_action_buffer()
        else:
            # replace the last (non-applied) action from the previous episode by the action that is going to be applied:
            if not self.last_act_on_reset:
                self.act_buf[-1] = self.default_action
        elt, info = self.interface.reset(seed=seed, options=options)
        if self.act_in_obs:
            elt = elt + list(self.act_buf)
        if self.obs_prepro_func:
            elt = self.obs_prepro_func(elt)
        elt = tuple(elt)
        if not self.time_initialized:
            self._initialize_time()
        if self.real_time:
            self._run_time_step(self.act_buf[-1])
        return elt, info

    def step(self, action):
        """Performs an environment step.

        Args:
            action: numpy.array: control value

        Returns:
            obs, rew, terminated, truncated, info

        CAUTION: this is a REAL-TIME step.
        This means step() needs to be called before the end of each time step.
        step() waits for the current time step to end, and returns the corresponding observation.
        In turn, the action is applied during the next time step.

        If you want to "pause" the environment at some point, use the wait() method.
        """
        if self.benchmark:
            self.bench.start_step_time()
        self._join_thread()
        self.current_step += 1
        self.act_buf.append(action)  # the action is always appended to the buffer
        if not self.real_time:
            self._run_time_step(action)
        if not self.running:
            raise RuntimeError("The episode is terminated or truncated. Call reset before step.")
        obs, rew, terminated, info = self._retrieve_obs_rew_terminated_info()
        truncated = (self.current_step >= self.ep_max_length) if not terminated else False
        done = (terminated or truncated)
        if not done:  # apply action only when not done
            self._run_time_step(action)
        else:
            self.running = False
            if self.wait_on_done:
                self.wait()
        if self.act_in_obs:
            obs = tuple((*obs, *tuple(self.act_buf),))
        if self.benchmark:
            self.bench.end_step_time()
        return obs, rew, terminated, truncated, info

    def stop(self):
        self._join_thread()

    def wait(self):
        """"Pauses" the environment.
        """
        self._join_thread()
        self.is_waiting = True
        self.interface.wait()

    def benchmarks(self):
        """Gets environment benchmarks.

        Caution: not compatible with render(join_thread=True)

        Returns:
            A dictionary containing the running averages and average deviations of important durations
        """
        assert self.benchmark, "The benchmark option is not set. Set benchmark=True in the configuration dictionary of the rtgym environment"
        return self.bench.get_benchmark_dict()

    def render(self, mode='human', join_thread=False):
        """Visually renders the current state of the environment.

        Args:
            mode: not used
            join_thread: set this to True if your render method performs unsafe operations.
                The render method of your interface is called outside the Real-Time Gym thread.
                Caution: when join_thread is True, render() is not compatible with benchmarks().
        """
        if join_thread:
            self._join_thread()
        self.interface.render()

    def set_default_action(self, default_action):
        """Changes the default action.

        Use this method right before calling reset() if you want the environment to send another default_action.
        This is useful when you want to maintain the real-time flow around the end of an episode.
        For instance, you may want to call set_default_action() with default_action as the action sent to step() right
        before the episode got terminated or truncated, because this action was never applied (thus, it will be applied
        by reset() - note however that this last action can be random unless you take special care).

        Note: alternatively, you can set the "last_act_on_reset" entry to True in the rtgym configuration.

        Args:
            default_action: numpy.array: new default action (make sure it complies with the action space)
        """
        self.default_action = default_action
