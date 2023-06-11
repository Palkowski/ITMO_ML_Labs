import numpy as np

# Question 1

probs = [1 / 2, 1 / 6, 1 / 3]


def entropy(p):
    h = 0
    for i in p:
        h += (i * np.log2(i))
    return -h


print('Question 1.1 answer:', round(entropy(probs), 3))
