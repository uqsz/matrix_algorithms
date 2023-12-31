import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import lil_matrix, csr_matrix, csgraph
import numpy as np
from collections import deque as Queue


def findIndex(a, x):
    return next((i for i, item in enumerate(a) if item[0] == x), -1)


class ReorderingSSM:
    def __init__(self, m):
        self.matrix = m
        self.n = len(m)

    def CuthillMckee(self):
        degrees = [sum(row) for row in self.matrix]

        Q = Queue()
        R = []
        notVisited = []

        notVisited = [(i, degrees[i]) for i in range(len(degrees))]

        while len(notVisited):

            minNodeIndex = min(range(len(notVisited)),
                               key=lambda i: notVisited[i][1])

            Q.append(notVisited[minNodeIndex][0])

            notVisited.pop(findIndex(notVisited, notVisited[Q[0]][0]))

            while Q:
                toSort = []
                v = Q.popleft()
                for i in range(self.n):
                    if (i != v and self.matrix[v][i] != 0 and findIndex(notVisited, i) != -1):
                        toSort.append(i)
                        notVisited.pop(findIndex(notVisited, i))

                toSort.sort(key=lambda x: degrees[x])
                Q.extend(toSort)
                R.append(v)

        return R

    def CM(self):
        cuthill = self.CuthillMckee()
        return cuthill

    def RCM(self):
        cuthill = self.CuthillMckee()
        return cuthill[::-1]

    def findIndex(a, x):
        return next((i for i, item in enumerate(a) if item[0] == x), -1)


# Driver Code
# num_rows = 10
# matrix = [[0.0] * num_rows for _ in range(num_rows)]

# matrix[0] = [0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
# matrix[1] = [1, 0, 0, 0, 1, 0, 1, 0, 0, 1]
# matrix[2] = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
# matrix[3] = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
# matrix[4] = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
# matrix[5] = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
# matrix[6] = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
# matrix[7] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
# matrix[8] = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
# matrix[9] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

# m = ReorderingSSM(matrix)

# r = m.RCM()

# print("Permutation order of objects:", r)


def generate_3d_grid_matrix(k):
    size = 2**(3*k)
    matrix = [[0] * size for _ in range(size)]

    for i in range(size):
        x, y, z = np.unravel_index(i, (2**k, 2**k, 2**k))
        neighbors = [
            np.ravel_multi_index((x + dx, y + dy, z + dz),
                                 (2**k, 2**k, 2**k), mode='wrap')
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if (dx != 0 or dy != 0 or dz != 0) and 0 <= x + dx < 2**k and 0 <= y + dy < 2**k and 0 <= z + dz < 2**k
        ]
        for neighbor in neighbors:
            matrix[i][neighbor] = 1

    return matrix


def plot_binary_matrix(matrix, x):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='binary', interpolation='none', vmin=0, vmax=1)

    plt.savefig("lab4/graphs/"+x)


def apply_permutation(matrix, permutation):
    n = len(matrix)
    permuted_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            permuted_matrix[i][j] = matrix[permutation[i]][permutation[j]]

    return permuted_matrix


for k in range(2, 5):
    matrix = generate_3d_grid_matrix(k)
    m = ReorderingSSM(matrix)
    r = m.RCM()
    permuted_matrix = apply_permutation(matrix, r)
    plot_binary_matrix(matrix, f"matrix{k}")
    plot_binary_matrix(permuted_matrix, f"permutated_matrix{k}")
