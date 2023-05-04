import numpy as np

# Question 3

data = np.array([[1, 3],
                 [1, -2],
                 [-2, -1]])
weights = np.array([[0.32],
                    [0.95]])
print('Question 3 answer:')
for i in range(0, len(np.matmul(data, weights))):
    print(f'z_{i + 1} =', round(np.matmul(data, weights)[i][0], 2))

# Question 4

a = np.array([2, 3])
b = np.array([-3, 2])
print('\nQuestion 4 answer:')
print('a \\dot b =', np.dot(a, b))
a_prime = a / np.linalg.norm(a)
b_prime = b / np.linalg.norm(b)
print(f'a\' = ({round(a_prime[0], 2)}, {round(a_prime[1], 2)})')
print(f'b\' = ({round(b_prime[0], 2)}, {round(b_prime[1], 2)})')

# Question 5

print('\nQuestion 5 answer:')
print('5, 5/3')

# Question 6

print('\nQuestion 6 answer:')
Z1 = np.array([[3.17],
               [-1.58],
               [-1.59]])
phi1 = np.array([[0.32], [0.95]]).T
print('Z1 * phi1 =\n', np.matmul(Z1, phi1).round(2))
