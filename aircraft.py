import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
        self.sojourn_times = []

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

    def execute(self, tend: float, Lambda:float=0.2) -> tuple[float, float, float]:
        """Perform simulation

        :param tend: Simulation duration [days]
        :return: Total operation, repair and waiting time
        """

        # Initialize operation times and tend
        self.tend = tend
        self.failure_times = [self.get_operation_time(Lambda) for _ in range(self.n)]
        self.total_operation_time = np.sum(self.failure_times)

        while self.now < self.tend:

            # Find first plane to fail and set now to its operational time
            broken_plane_idx, self.now = self.get_broken_plane()

            # Find time to repair failed plane and update now to end of repair
            repair_time = self.get_repair_time()
            self.total_repair_time += repair_time
            # make a list of lists with repair time and week number
            self.sojourn_times.append([repair_time, self.now // 7]) 
            #update now to end of repair
            self.now += repair_time

            # Set new operation time for plane (0 if now = end)
            operation_time = self.get_operation_time(Lambda)
            self.total_operation_time += operation_time
            self.failure_times[broken_plane_idx] = self.now + operation_time

            # Bring rest of planes to present
            while min(self.failure_times) < self.now:

                # Get index and time of failure for next plane to break
                broken_plane_idx, break_time = self.get_broken_plane()

                # Find waiting time for broken plane
                waiting_time = self.get_waiting_time(break_time)
                self.total_waiting_time += waiting_time
                self.sojourn_times.append([waiting_time, self.now // 7])

                

                if self.now <= self.tend:

                    # Find time to repair plane and update now to end of repair
                    repair_time = self.get_repair_time()
                    self.total_repair_time += repair_time
                    self.now += repair_time
                    # add repair time to waiting time in sojourn_times
                    self.sojourn_times[-1][0] += repair_time

                    # Find new operation time for plane
                    operation_time = self.get_operation_time(Lambda)
                    self.total_operation_time += operation_time
                    self.failure_times[broken_plane_idx] = self.now + \
                        operation_time

        assert np.abs((self.total_operation_time + self.total_waiting_time +
                       self.total_repair_time) - (self.n * self.tend)) < 1e-5

        return (self.total_operation_time,
                self.total_repair_time, self.total_waiting_time)


def exercises(planes: int, duration: float, runs: int) -> None:

    cases = [Simulate(planes) for _ in range(runs)]
    data = np.array([case.execute(duration) for case in cases])

    print(cases[0].sojourn_times)

    downtime = data[:, 1:].sum(axis=1)
    mu = downtime.sum() / runs # Expected value for downtime

    variance = (downtime - mu)**2/(runs)
    print(f"Standard deviation: {round(np.sqrt(variance.sum()),3)}")

    print(f"Standard Error: {round(np.sqrt(variance.sum()/runs),3)}")

    print(f"Expected downtime: {round(mu,3)}")

    larger = 0.
    for time in downtime:
        if time > 4:
            larger += 1

    print("Expected")

    probability = larger / runs # Probability of downtime > 4

    print(f"P(X > 4) = {probability}")

    print(f"Standard Error of P(X > 4): {round(np.sqrt(probability*(1-probability)/runs),3)}")

    return None

def sojourn_time_plot(sojourn_time_list:list, lag:int=1):

    x = sojourn_time_list[:(len(sojourn_time_list)-lag)]
    y = sojourn_time_list[lag:]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(x, y, s=1)
    plt.show()

def calculate_autocorrelation(my_list:list, lag:int=1):
    
        x = my_list[:(len(my_list)-lag)]
        y = my_list[lag:]
    
        return np.corrcoef(x, y)[0][1]

def calculate_expected_value_of_list(my_list:list):
    return sum(my_list)/len(my_list)

def calculate_expected_value_of_list_batch(my_list:list, batch_size:int):
    batched_list = [my_list[i:i+batch_size] for i in range(0, len(my_list), batch_size)]
    expected_values = [calculate_expected_value_of_list(batch) for batch in batched_list]
    return calculate_expected_value_of_list(expected_values)

def calculate_standard_deviation_of_list(my_list:list):
    return np.std(my_list) 

def calculate_standard_deviation_of_list_batch(my_list:list, batch_size:int):  
    batched_list = [my_list[i:i+batch_size] for i in range(0, len(my_list), batch_size)]
    standard_deviations =  [calculate_standard_deviation_of_list(batch) for batch in batched_list]

    print('std: ', calculate_standard_deviation_of_list(standard_deviations))

    return calculate_expected_value_of_list(standard_deviations)



def assignment_4_1(no_of_planes:int, weeks:int):

    t_end = weeks*7

    week_sojourn_times_1 = [0]*len(range(weeks))
    week_sojourn_times_3 = [0]*len(range(weeks))

    simulation_case_1 = Simulate(no_of_planes)
    simulation_case_1.execute(t_end, 0.1)

    sojourn_times_1 = np.array(simulation_case_1.sojourn_times).swapaxes(0, 1)[0]

    

    simulation_case_3 = Simulate(no_of_planes)
    simulation_case_3.execute(t_end, 0.3)

    sojourn_times_3 = np.array(simulation_case_3.sojourn_times).swapaxes(0, 1)[0]

    

    # sojourn_time_per_week = [0]*len(range(weeks))

    # for week in range(weeks):
    #     sojourn_time_per_repair = simulation_case.sojourn_times[week][0]
    #     sojourn_time_per_week[week] += sojourn_time_per_repair

    for sojourn_time_per_repair, week in simulation_case_1.sojourn_times:
        if week < weeks:
            # print(f"Repair time: {sojourn_time_per_repair} week: {week}")
            week_sojourn_times_1[int(week)] += sojourn_time_per_repair

    for sojourn_time_per_repair, week in simulation_case_3.sojourn_times:
        if week < weeks:
            # print(f"Repair time: {sojourn_time_per_repair} week: {week}")
            week_sojourn_times_3[int(week)] += sojourn_time_per_repair

    # print(week_sojourn_times_1)
    # print(week_sojourn_times_3)

    # part a
    sojourn_time_plot(sojourn_times_1, 1)
    sojourn_time_plot(sojourn_times_3, 1)

    # part b
    part_b_table = pd.DataFrame(columns=['lag', 'autocorr_sojourn_times_1', 'autocorr_sojourn_times_3', 'autocorr_week_1', 'autocorr_week_3'])

    for lag in range(1, 9):
        autocorr_sojourn_times_1 = calculate_autocorrelation(sojourn_times_1, lag)
        autocorr_sojourn_times_3 = calculate_autocorrelation(sojourn_times_3, lag)
        autocorr_week_1 = calculate_autocorrelation(week_sojourn_times_1, lag)
        autocorr_week_3 = calculate_autocorrelation(week_sojourn_times_3, lag)

        part_b_table.loc[lag] = [lag, autocorr_sojourn_times_1, autocorr_sojourn_times_3, autocorr_week_1, autocorr_week_3]

    print(part_b_table)

    # part c

    batch_size_min = 2
    batch_size = 25

    part_c_table = pd.DataFrame(columns=[
        'data', 'expected_value', 'standard_deviation', 
        'expected_value_batch', 'standard_deviation_batch', 
        'expected_value_batch_min', 'standard_deviation_batch_min'])
    
    part_c_table['data'] = [
        'sojourn_times_1', 'sojourn_times_3', 'week_sojourn_times_1', 'week_sojourn_times_3']
    part_c_table['expected_value'] = [
        calculate_expected_value_of_list(sojourn_times_1), 
        calculate_expected_value_of_list(sojourn_times_3), 
        calculate_expected_value_of_list(week_sojourn_times_1), 
        calculate_expected_value_of_list(week_sojourn_times_3)]
    part_c_table['standard_deviation'] = [
        calculate_standard_deviation_of_list(sojourn_times_1), 
        calculate_standard_deviation_of_list(sojourn_times_3), 
        calculate_standard_deviation_of_list(week_sojourn_times_1), 
        calculate_standard_deviation_of_list(week_sojourn_times_3)]
    part_c_table['expected_value_batch'] = [
        calculate_expected_value_of_list_batch(sojourn_times_1, batch_size), 
        calculate_expected_value_of_list_batch(sojourn_times_3, batch_size), 
        calculate_expected_value_of_list_batch(week_sojourn_times_1, batch_size), 
        calculate_expected_value_of_list_batch(week_sojourn_times_3, batch_size)]
    part_c_table['standard_deviation_batch'] = [
        calculate_standard_deviation_of_list_batch(sojourn_times_1, batch_size), 
        calculate_standard_deviation_of_list_batch(sojourn_times_3, batch_size), 
        calculate_standard_deviation_of_list_batch(week_sojourn_times_1, batch_size), 
        calculate_standard_deviation_of_list_batch(week_sojourn_times_3, batch_size)]
    part_c_table['expected_value_batch_min'] = [
        calculate_expected_value_of_list_batch(sojourn_times_1, batch_size_min), 
        calculate_expected_value_of_list_batch(sojourn_times_3, batch_size_min), 
        calculate_expected_value_of_list_batch(week_sojourn_times_1, batch_size_min), 
        calculate_expected_value_of_list_batch(week_sojourn_times_3, batch_size_min)]
    part_c_table['standard_deviation_batch_min'] = [
        calculate_standard_deviation_of_list_batch(sojourn_times_1, batch_size_min), 
        calculate_standard_deviation_of_list_batch(sojourn_times_3, batch_size_min), 
        calculate_standard_deviation_of_list_batch(week_sojourn_times_1, batch_size_min), 
        calculate_standard_deviation_of_list_batch(week_sojourn_times_3, batch_size_min)]

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(part_c_table)


if __name__ == "__main__":

    # exercises(10, 7*500, 1)
    assignment_4_1(10, 500)
