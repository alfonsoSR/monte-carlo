import numpy as np


class Simulate:
    """Simulate aircraft maintenance problem

    :param n: Number of aircraft
    """

    def __init__(self, n: int) -> None:

        self.n = n
        self.tend = 0.

        self.now = 0.
        self.total_operation_time = 0.
        self.total_waiting_time = 0.
        self.total_repair_time = 0.

        self.failure_times = [0. for _ in range(self.n)]

        return None

    def get_operation_time(self, L: float = 0.2) -> float:
        """Calculate operation time for a given plane.

        Operation time (top) is calculated using an exponential distribution 
        with parameter L.

        If top + now > tend, we set top = tend - now.

        :param L: Lambda parameter of exponential distribution
        """
        _operation_time = np.random.exponential(1. / L)

        if _operation_time + self.now > self.tend:
            return self.tend - self.now
        else:
            return _operation_time

    def get_repair_time(self, mu: float = 0.25, std: float = 0.05) -> float:
        """Calculate repair time for a given aircraft.

        Repair time (tr) is calculated using a normal distribution with
        expected value mu and variance std^2.

        If tr + now > tend, we set tr = tend - now

        :param mu: Expected value
        :param std: Standard deviation [days]
        """
        _repair_time = np.random.normal(mu, std)

        if _repair_time + self.now >= self.tend:
            return self.tend - self.now
        else:
            return _repair_time

    def get_waiting_time(self, failure_time: float) -> float:
        """Calculate waiting time for a given plane

        :param failure_time: Time of failure for waiting plane
        """
        return self.now - failure_time

    def get_broken_plane(self) -> tuple[int, float]:
        """Find index and operational time of next plane to fail"""

        _time = min(self.failure_times)
        _plane = self.failure_times.index(_time)
        return _plane, _time

    def __call__(self, tend: float) -> tuple[float, float, float]:
        """Perform simulation

        :param tend: Simulation duration [days]
        :return: Total operation, repair and waiting time 
        """

        # Initialize operation times and tend
        self.tend = tend
        self.failure_times = [self.get_operation_time() for _ in range(self.n)]

        while self.now < self.tend:

            # Find first plane to fail and set now to its operational time
            broken_plane_idx, self.now = self.get_broken_plane()

            # Find time to repair failed plane and update now to end of repair
            repair_time = self.get_repair_time()
            self.total_repair_time += repair_time
            self.now += repair_time

            # Set new operation time for plane (0 if now = end)
            operation_time = self.get_operation_time()
            self.total_operation_time += operation_time
            self.failure_times[broken_plane_idx] = self.now + operation_time

            # Bring rest of planes to present
            while min(self.failure_times) < self.now:

                # Get index and time of failure for next plane to break
                broken_plane_idx, break_time = self.get_broken_plane()

                # Find waiting time for broken plane
                waiting_time = self.get_waiting_time(break_time)
                self.total_waiting_time += waiting_time

                if self.now <= self.tend:

                    # Find time to repair plane and update now to end of repair
                    repair_time = self.get_repair_time()
                    self.total_repair_time += repair_time
                    self.now += repair_time

                    # Find new operation time for plane
                    operation_time = self.get_operation_time()
                    self.total_operation_time += operation_time
                    self.failure_times[broken_plane_idx] = self.now + \
                        operation_time

        assert (self.total_operation_time + self.total_waiting_time +
                self.total_waiting_time) - self.n * self.tend < 1e-5

        return (self.total_operation_time,
                self.total_repair_time, self.total_waiting_time)


if __name__ == "__main__":

    maintenance = Simulate(10)
    maintenance(7)
