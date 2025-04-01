import time
from threading import Lock


class Benchmark:
    """Benchmarks of the durations of main operations.

    A running average of the mean and average deviation are provided for each duration.
    The results are returned as a dictionary of (average, average_deviation) entries.
    """
    def __init__(self, run_avg_factor=0.1):
        self.run_avg_factor = run_avg_factor
        self._b_lock = Lock()

        # times:
        self.__start_step_time = None
        self.__end_join_time = None
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
        self._b_lock.acquire()
        res = {
            "time_step_duration": (self.time_step_duration, self.time_step_duration_dev),
            "step_duration": (self.step_duration, self.step_duration_dev),
            "join_duration": (self.join_duration, self.join_duration_dev),
            "inference_duration": (self.inference_duration, self.inference_duration_dev),
            "send_control_duration": (self.send_control_duration, self.send_control_duration_dev),
            "retrieve_obs_duration": (self.retrieve_obs_duration, self.retrieve_obs_duration_dev),
        }
        self._b_lock.release()
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
        self._b_lock.acquire()
        now = time.perf_counter()
        if self.__end_step_time is not None:
            self.inference_duration, self.inference_duration_dev = self.running_average(new_val=now - self.__end_step_time, old_avg=self.inference_duration, old_avg_dev=self.inference_duration_dev)
        self.__start_step_time = now
        self._b_lock.release()

    def end_join_time(self):
        """after join().
        """
        self._b_lock.acquire()
        now = time.perf_counter()
        if self.__start_step_time is not None:
            self.join_duration, self.join_duration_dev = self.running_average(new_val=now - self.__start_step_time, old_avg=self.join_duration, old_avg_dev=self.join_duration_dev)
        self.__end_join_time = now
        self._b_lock.release()

    def end_step_time(self):
        """before return.
        """
        self._b_lock.acquire()
        now = time.perf_counter()
        if self.__start_step_time is not None:
            self.step_duration, self.step_duration_dev = self.running_average(new_val=now - self.__start_step_time, old_avg=self.step_duration, old_avg_dev=self.step_duration_dev)
        self.__end_step_time = now
        self._b_lock.release()

    def start_time_step_time(self):
        """before run_time_step.
        """
        self._b_lock.acquire()
        now = time.perf_counter()
        if self.__start_time_step_time is not None:
            self.time_step_duration, self.time_step_duration_dev = self.running_average(new_val=now - self.__start_time_step_time, old_avg=self.time_step_duration, old_avg_dev=self.time_step_duration_dev)
        self.__start_time_step_time = now
        self._b_lock.release()

    def start_retrieve_obs_time(self):
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__start_retrieve_obs_time = now
        self._b_lock.release()

    def end_retrieve_obs_time(self):
        self._b_lock.acquire()
        now = time.perf_counter()
        if self.__start_retrieve_obs_time is not None:
            self.retrieve_obs_duration, self.retrieve_obs_duration_dev = self.running_average(new_val=now - self.__start_retrieve_obs_time, old_avg=self.retrieve_obs_duration, old_avg_dev=self.retrieve_obs_duration_dev)
        self.__end_retrieve_obs_time = now
        self._b_lock.release()

    def end_send_control_time(self):
        self._b_lock.acquire()
        now = time.perf_counter()
        if self.__start_time_step_time is not None:
            self.send_control_duration, self.send_control_duration_dev = self.running_average(new_val=now - self.__start_time_step_time, old_avg=self.send_control_duration, old_avg_dev=self.send_control_duration_dev)
        self.__end_send_control_time = now
        self._b_lock.release()


class TraceBenchmark(Benchmark):
    def __init__(self, run_avg_factor=0.1):
        super().__init__(run_avg_factor=run_avg_factor)

        # histories:
        self.__start_step_time = []
        self.__end_join_time = []
        self.__end_step_time = []
        self.__start_retrieve_obs_time = []
        self.__end_retrieve_obs_time = []
        self.__start_time_step_time = []
        self.__end_send_control_time = []

    def get_benchmark_dict(self):
        """Returns A dict containing all time traces.
        """
        from copy import deepcopy
        import numpy as np

        self._b_lock.acquire()
        res = {
            "times": {
                "start_step": deepcopy(self.__start_step_time),
                "end_join": deepcopy(self.__end_join_time),
                "end_step": deepcopy(self.__end_step_time),
                "start_retrieve_obs": deepcopy(self.__start_retrieve_obs_time),
                "end_retrieve_obs": deepcopy(self.__end_retrieve_obs_time),
                "start_time_step": deepcopy(self.__start_time_step_time),
                "end_send_control": deepcopy(self.__end_send_control_time),
            },
            "durations": {
                "time_step_duration": None,
                "step_duration": None,
                "join_duration": None,
                "inference_duration": None,
                "send_control_duration": None,
                "retrieve_obs_duration": None,
            }
        }
        self._b_lock.release()
        return res

    def start_step_time(self):
        """before join().
        """
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__start_step_time.append(now)
        self._b_lock.release()

    def end_join_time(self):
        """after join().
        """
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__end_join_time.append(now)
        self._b_lock.release()

    def end_step_time(self):
        """before return.
        """
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__end_step_time.append(now)
        self._b_lock.release()

    def start_time_step_time(self):
        """before run_time_step.
        """
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__start_time_step_time.append(now)
        self._b_lock.release()

    def start_retrieve_obs_time(self):
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__start_retrieve_obs_time.append(now)
        self._b_lock.release()

    def end_retrieve_obs_time(self):
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__end_retrieve_obs_time.append(now)
        self._b_lock.release()

    def end_send_control_time(self):
        self._b_lock.acquire()
        now = time.perf_counter()
        self.__end_send_control_time.append(now)
        self._b_lock.release()