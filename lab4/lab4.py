import copy
import random
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import lil_matrix, csr_matrix, csgraph
import numpy as np
from collections import deque as Queue, deque

from sklearn.utils.extmath import randomized_svd


class Node:
    def __init__(self, size, rank=0):
        self.children = []
        self.parent = None
        self.size = size
        self.rank = rank
        self.sv = []
        self.U = None
        self.V = None

    def append(self, node):
        self.children.append(node)
        node.parent = self


def create_tree(t_min, t_max, s_min, s_max, r, eps):
    global ax
    global matrix_to_svd
    U, D, V = randomized_svd(
        matrix_to_svd[t_min:t_max, s_min:s_max], n_components=r + 1, random_state=0)
    if len(D) <= r or D[r] < eps:
        draw_black((t_min, s_min), (t_max, s_max), ax)
        v = compress_matrix(t_min, t_max, s_min, s_max, U, D, V, r)

    else:
        draw_cross_with_square((t_min, s_min), (t_max, s_max), ax)
        v = Node((t_min, t_max, s_min, s_max))
        t_new_max = t_min + (t_max - t_min) // 2
        s_new_max = s_min + (s_max - s_min) // 2

        v.append(create_tree(t_min, t_new_max, s_min, s_new_max, r, eps))
        v.append(create_tree(t_min, t_new_max, s_new_max, s_max, r, eps))
        v.append(create_tree(t_new_max, t_max, s_min, s_new_max, r, eps))
        v.append(create_tree(t_new_max, t_max, s_new_max, s_max, r, eps))
    return v


def compress_matrix(t_min, t_max, s_min, s_max, U, D, V, r):
    global matrix_to_svd
    if np.all(matrix_to_svd[t_min:t_max, s_min:s_max] == 0):
        v = Node((t_min, t_max, s_min, s_max))
        v.U = np.zeros((t_max - t_min, s_max - s_min))
        v.V = np.zeros((t_max - t_min, s_max - s_min))
        return v
    v = Node((t_min, t_max, s_min, s_max), r)
    v.sv = D[:r+1]
    v.U = U[:, :r + 1]
    v.V = np.diag(D[:r + 1]) @ V[:r + 1, :]
    return v


def draw_cross_with_square(left_top, right_bottom, ax):

    center_x = (left_top[0] + right_bottom[0]) / 2
    center_y = (left_top[1] + right_bottom[1]) / 2
    width = right_bottom[0] - left_top[0]
    height = right_bottom[1] - left_top[1]
    rect = patches.Rectangle(left_top, width, height,
                             linewidth=0.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    cross_size = min(width, height)
    ax.plot([center_x - cross_size / 2, center_x + cross_size / 2],
            [center_y, center_y], color='black', linewidth=0.5)
    ax.plot([center_x, center_x], [center_y - cross_size / 2,
            center_y + cross_size / 2], color='black', linewidth=0.5)


def draw_black(left_top, right_bottom, ax):
    width = right_bottom[0] - left_top[0]
    height = right_bottom[1] - left_top[1]
    rect = patches.Rectangle(left_top, width*0.2, height,
                             linewidth=0.5, color='black', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(right_bottom, -width, -height*0.2,
                             linewidth=0.5, color='black', facecolor='none')
    ax.add_patch(rect)


def show_matrices(name, k):  # rysownik
    global matrix_to_svd
    n = len(matrix_to_svd)
    U, D, V = randomized_svd(matrix_to_svd, n_components=n, random_state=0)
    global ax
    fig, ax = plt.subplots(figsize=(10.24, 10.24))
    v = create_tree(0, n, 0, n, 1, D[n-1])

    ax.invert_yaxis()
    ax.set_title(f'H-macierz: k={k}, b=1, i={n-1}')
    plt.savefig(f'graphs/{name}')
    # plt.show()
    plt.clf()


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

    def minimumDegree(self):
        degrees = [sum(row) for row in self.matrix]
        notVisited = [i for i in range(self.n)]
        matrixCopy = copy.deepcopy(self.matrix)
        res = []

        for k in range(self.n):
            notVisited.sort(key=lambda x: (degrees[x], x))
            p = notVisited.pop(0)
            res.append(p)
            for i in range(self.n):
                if matrixCopy[p][i] != 0:
                    matrixCopy[p][i] = 0
                    matrixCopy[i][p] = 0
                    degrees[i] -= 1
                    for j in range(i + 1, self.n):
                        if matrixCopy[p][j] != 0 and matrixCopy[i][j] == 0:
                            matrixCopy[i][j] = 1
                            matrixCopy[j][i] = 1
                            degrees[i] += 1
                            degrees[j] += 1
        return res

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
    eps = np.finfo(float).eps

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
            matrix[i][neighbor] = random.uniform(10 ** -8 + eps, 1.0 - eps)

    return np.array(matrix)


def plot_binary_matrix(matrix, name):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='binary', interpolation='none', vmin=0, vmax=1)
    # plt.show()
    plt.savefig("graphs/"+name)


def apply_permutation(matrix, permutation):
    n = len(matrix)
    permuted_matrix = np.zeros((n, n), dtype=type(matrix[0][0]))

    for i in range(n):
        for j in range(n):
            permuted_matrix[i, j] = matrix[permutation[i]][permutation[j]]

    return permuted_matrix


def test(k):
    global matrix_to_svd
    matrix = generate_3d_grid_matrix(k)
    matrix_to_svd = copy.deepcopy(matrix)
    plot_binary_matrix(matrix, f"matrix/matrix{k}")
    show_matrices(f"/matrix/svd{k}", k)

    m = ReorderingSSM(matrix)
    r = m.minimumDegree()
    permuted_matrix = apply_permutation(matrix, r)
    matrix_to_svd = copy.deepcopy(permuted_matrix)
    plot_binary_matrix(permuted_matrix, f"minimum_degree/minimum_degree{k}")
    show_matrices(f"minimum_degree/svd{k}", k)

    r = m.CM()
    permuted_matrix = apply_permutation(matrix, r)
    matrix_to_svd = copy.deepcopy(permuted_matrix)
    plot_binary_matrix(permuted_matrix, f"CM/CM{k}")
    show_matrices(f"CM/svd{k}", k)

    r = m.RCM()
    permuted_matrix = apply_permutation(matrix, r)
    matrix_to_svd = copy.deepcopy(permuted_matrix)
    plot_binary_matrix(permuted_matrix, f"RCM/RCM{k}")
    show_matrices(f"RCM/svd{k}", k)

    # plot_binary_matrix(matrix, f"matrix{k}")
    # m = ReorderingSSM(matrix)
    #
    # r = m.CM()
    # permuted_matrix = apply_permutation(matrix, r)
    # plot_binary_matrix(permuted_matrix, f"permutated_matrix{k}")
    #
    # r_rev = m.RCM()
    # reversed_permuted_matrix = apply_permutation(matrix, r_rev)
    # plot_binary_matrix(reversed_permuted_matrix,
    #                    f"reversed_permutated_matrix{k}")
    #
    # r = m.minimumDegree()
    # minimum_degree_matrix = apply_permutation(matrix, r)
    # plot_binary_matrix(minimum_degree_matrix,
    #                    f"minimum_degree_matrix{k}")

#
# for k in range(2, 5):
#     matrix = generate_3d_grid_matrix(k)
#     plot_binary_matrix(matrix, f"matrix{k}")
#     m = ReorderingSSM(matrix)
#
#     r = m.CM()
#     permuted_matrix = apply_permutation(matrix, r)
#     plot_binary_matrix(permuted_matrix, f"permutated_matrix{k}")
#
#     r_rev = m.RCM()
#     reversed_permuted_matrix = apply_permutation(matrix, r_rev)
#     plot_binary_matrix(reversed_permuted_matrix,
#                        f"reversed_permutated_matrix{k}")
#
#     r = m.minimumDegree()
#     minimum_degree_matrix = apply_permutation(matrix, r)
#     plot_binary_matrix(minimum_degree_matrix,
#                        f"minimum_degree_matrix{k}")

# matrix = [[0, 1, 1, 1, 1],
#           [1, 0, 0, 0, 0],
#           [1, 0, 0, 0, 0],
#           [1, 0, 0, 0, 0],
#           [1, 0, 0, 0, 0]]
# m = ReorderingSSM(matrix)
# r = m.minimumDegree()
# x = apply_permutation(matrix, r)
# print(r)
# print()
# print(x)
# print()
# print(matrix)
# plot_binary_matrix(x, "minimum_degree_matrix2")


if __name__ == "__main__":
    for i in range(2, 5):
        test(i)
