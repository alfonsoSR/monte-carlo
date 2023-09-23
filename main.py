import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ex1(data: pd.DataFrame):
    """Probability of breach

    Rationale: The probability of a breach is the ratio between the number of
    breaches in our simulation and the total number of events
    """

    breaches = data["B(reach)"]
    assert breaches.shape[0] == 1000
    return np.sum(breaches) / breaches.shape[0]


def ex2(data: pd.DataFrame, failed_edge: str):
    """Probability of a breach knowing that given edge failed

    Rationale: Similarly to what we did to calculate the probability of a
    breach, we can calculate P(B|e7) by dividing the number of breaches when
    e7 failed by the total number of events when e7 failed."""

    events = []

    for fail, breach in zip(data[failed_edge], data["B(reach)"]):
        if fail == 1:
            events.append(breach)

    return np.sum(events) / len(events)


def make_plots(data: pd.DataFrame, func):

    P = [func(data, f"e{idx}") for idx in range(1, 23)]
    e_list = [f"e{idx}" for idx in range(1, 23)]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(e_list, P)
    ax.set_title(
        "Probability of an edge having failed knowing that a Breach occured")
    ax.set_xlabel("Failing edge")
    ax.set_ylabel(r"$P(e_i \mid B)$")
    plt.show()


def ex3(data: pd.DataFrame, failed_edge: str):
    """Probability of e4 having failed knowing that a breach occured

    Rationale: Having functions to calculate the probability of a breach and
    the conditional probability of a breach knowing that a certain edge fails,
    we can use Bayes' rule to calculate this conditional probability. We just
    need to calculate the probability of e4 failing, which is the number of
    times e4 failed over the total amount of events."""

    p_ei = np.sum(data[failed_edge]) / data[failed_edge].shape[0]
    p_ei_B = ex2(data, failed_edge)
    p_B = ex1(data)

    return p_ei * p_ei_B / p_B


def ex4(data: pd.DataFrame, k: int):
    """Probability of breach knowing that k edges failed"""

    n_fails = np.sum(np.array(data).swapaxes(0, 1)[:-1].swapaxes(0, 1), axis=1)

    breaches = []

    for fails, breach in zip(n_fails, data["B(reach)"]):
        if fails == k:
            breaches.append(breach)

    if len(breaches) == 0:
        return 0
    else:
        return np.sum(breaches) / len(breaches)


def obvious(data: pd.DataFrame):

    fails = [idx for idx in range(1, 23)]
    P = [ex4(data, k) for k in fails]

    plt.bar(fails, P)
    plt.show()


if __name__ == "__main__":

    data = pd.read_excel("intruder-results.xlsx", "Sheet1")

    print(ex1(data))
    print(ex2(data, "e7"))
    print(ex3(data, "e4"))
    print(ex4(data, 6))
    obvious(data)
