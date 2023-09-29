# This version: September 27th, 2022
# Programmer: Ludolf Meester, Etienne Guichard

import numpy as np
import random
from math import comb

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

def binomial_distribution(n: int, k: int, q: float) -> float:
    """
    Calculate the probability of k successes in n independent Bernoulli trials
    with probability of success q.

    :param n: Number of trials
    :param k: Number of successes
    :param q: Probability of success
    :return: Probability of k successes in n trials
    """
    return comb(n, k) * (q ** k) * ((1 - q) ** (n - k))

# Reading data and setup list of edges
filename = 'airport-graph.txt'
# Conventions the input file must satisfy:
#     two integers per line, denoting two nodes connected by an edge
#     source node has number 0; highest number is the terminal node
with open(filename) as file_object:
    lines = file_object.readlines()
edges = [tuple(map(int, line.split())) for line in lines] 
nedges = len(edges)
print("List of", nedges, "edges read from", filename, ":")
# print(edges)
# determine t(erminal)node
tnode = 0
tnode = max(max(edges))
# print("Terminal node:",tnode)

# Function sysfail
def sysfail(failed, tnode, maxpathlen):
    # input:
    #    failed: a list of tuples (n1,n2), representing the failed edges
    #    tnode: terminal node number
    #    maxpathlen: length of longest path from 0 to tnode
    breach = 0
    # construct the incidence matrix:
    Imat = np.zeros((tnode+1,tnode+1),dtype='int')
    failedarr = np.array(failed)
    if failedarr.shape[0] != 0:
        Imat[failedarr[:,0],failedarr[:,1]] = Imat[failedarr[::-1,1],failedarr[::-1,0]]= 1
    # if node 0 or tnode is isolated a breach is impossible
    if np.min([np.max(Imat[0,:]),np.max(Imat[:,tnode])]) == 0:
        return breach
    A = Imat
    #  if i-th power of Imat has nonzero (0,tnode) element, there is a
    # path of length i from 0 -> tnode: a breach
    for i in range(1,maxpathlen):
        if A[0,tnode]>0:
            breach = 1
            break
        A = np.matmul(A,Imat)
    return breach

# Simulation
seedval = 3875663
np.random.seed(seedval)
nrep = 1000
q = 0.2
maxpathlen = 22


parameters = [ [5,200,0], [6,200,0], [7,200,0], [8,200,0], [9,200,0] ]

for parameter in parameters:

    breaches = 0

    #print("Simulating", parameter[1], "replications for q=", q, ", ", parameter[0], "number of failed edges and seedvalue", seedval)
    for i in range(parameter[1]):
        # unif = np.random.random(nedges)
        # create list of failed edges:
        failed = get_failed_edges(0,21,parameter[0])
        breaches += sysfail(failed, tnode, maxpathlen)
    parameter[2] = breaches/parameter[1]
        
    #print("There were", breaches, "security breaches. Estimated P(breach)=",parameter[2])

print(parameters)

p_b = 0

for parameter in parameters:
    p_b += parameter[2] * binomial_distribution(nedges,parameter[0],q)

print("P(breach) = ", p_b)