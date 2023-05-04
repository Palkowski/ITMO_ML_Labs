import numpy as np

# Question 3

print('Question 3 answer:')
x_1 = np.array([2, 3, 4])
x_2 = np.array([-1, 3, 5])
print('dot product:', np.dot(x_1, x_2))


def f(x1, x2, x3):
    return 5 + 3 * x1 - 4 * x2 + 2 * x3


print('(1, -2, 3) class:', f(1, -2, 3))
