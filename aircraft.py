import numpy as np
import matplotlib.pyplot as plt


def get_failure_time(tend: float, L: float = 0.2):
    """Calculate time till next failure

    Uses exponential distribution with parameter L"""

    val = np.random.exponential(1. / L)

    if val >= tend:
        return tend
    else:
        return val


def get_repair_time(mu: float = 0.25, std: float = 0.05) -> float:
    """Calculate repair time for a given plane

    Uses normal distribution"""

    return np.random.normal(mu, std)


def simulate(n: int = 3, tend: float = 7.):

    now = 0.
    time_to_failure = [get_failure_time(tend) for _ in range(n)]
    total_operational_time = np.sum(time_to_failure)
    total_repair_time = 0.
    total_waiting_time = 0.

    while now < tend:

        i = time_to_failure.index(min(time_to_failure))

        _repair_time = get_repair_time()
        if now + _repair_time >= tend:
            total_repair_time += tend - now
            break
        else:
            total_repair_time += _repair_time
        now = time_to_failure[i] + _repair_time         #
        _new_failure_time = get_failure_time(tend)
        if now + _new_failure_time >= tend:
            total_operational_time += tend - now
        else:
            total_operational_time += _new_failure_time
        time_to_failure[i] = now + _new_failure_time
        if time_to_failure[i] > tend:
            time_to_failure[i] = tend

        while min(time_to_failure) < now:

            i = time_to_failure.index(min(time_to_failure))
            total_waiting_time += now - time_to_failure[i]
            _repair_time = get_repair_time()
            if now + _repair_time >= tend:
                total_repair_time += tend - now
                break
            else:
                total_repair_time += _repair_time
            now += _repair_time                                     #
            _new_failure_time = get_failure_time(tend)
            if now + _new_failure_time >= tend:
                total_operational_time += tend - now
            else:
                total_operational_time += _new_failure_time
            time_to_failure[i] = now + _new_failure_time
            if time_to_failure[i] > 7:
                time_to_failure[i] = 7.

    assert (
        total_operational_time + total_repair_time + total_waiting_time ==
        tend * n
    )


if __name__ == "__main__":

    simulate()
