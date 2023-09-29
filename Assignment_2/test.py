import random

def generate_random_integers_within_range(start, end, k):
    """
    Generates a list of k different random integers within the specified range [start, end].

    Args:
        start (int): The lower bound of the range (inclusive).
        end (int): The upper bound of the range (inclusive).
        k (int): The number of random integers to generate.

    Returns:
        list: A list of k different random integers within the specified range.
    """
    # Check if the range is valid
    if end < start:
        raise ValueError("End value must be greater than or equal to start value")

    # Generate a list of k different random integers
    random_integers = random.sample(range(start, end + 1), k)
    
    return random_integers

def get_failed_edges(start, end, k):
    """
    Returns a list of k randomly selected edges from the range [start, end) of the global list `edges`.

    Args:
        start (int): The start index of the range (inclusive).
        end (int): The end index of the range (exclusive).
        k (int): The number of edges to select.

    Returns:
        List[Tuple[int, int]]: A list of k randomly selected edges from the range [start, end) of the global list `edges`.
    """
    failed_indexes = (generate_random_integers_within_range(start,end,k))
    failed_edges = [edges[i] for i in failed_indexes]
    return failed_edges

filename = 'airport-graph.txt'
# Conventions the input file must satisfy:
#     two integers per line, denoting two nodes connected by an edge
#     source node has number 0; highest number is the terminal node
with open(filename) as file_object:
    lines = file_object.readlines()
edges = [tuple(map(int, line.split())) for line in lines] 

parameters = [[5,200,0], [6,200,0], [7,200,0], [8,200,0], [9,200,0]]


print(parameters[0][0])

