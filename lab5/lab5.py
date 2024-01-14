import copy
import random
import time

import numpy
import numpy as np
from matplotlib import pyplot as plt, patches
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
    global A
    U, D, V = randomized_svd(
        A[t_min:t_max, s_min:s_max], n_components=r+1, random_state=0)
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
    global A
    if np.all(A[t_min:t_max, s_min:s_max] == 0):
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

    global ax

    ax.invert_yaxis()
    ax.set_title(f'H-macierz: k={k}')
    plt.savefig(f'lab5/graphs/{name}')
    plt.clf()


def decompress(B, v):
    if v.children == []:
        B[v.size[0]: v.size[1], v.size[2]: v.size[3]] = v.U @ v.V
        return B
    for child in v.children:
        B = decompress(B, child)
    return B


def plot_binary_matrix(matrix, name):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='binary', interpolation='none', vmin=0, vmax=1)
    plt.title(f'Macierz: k={k}')
    plt.savefig("lab5/graphs/"+name)


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
            matrix[i][neighbor] = 1
        matrix[i][i] = 1
    return np.array(matrix)


def matrix_vector_mul(v, X):
    if v.children == []:
        if v.rank == 0:
            return np.zeros((v.size[1] - v.size[0], len(X[0])))
        return v.U @ (v.V @ X)
    rows = len(X)
    X1 = X[:rows//2, :]
    X2 = X[rows//2: rows, :]
    Y_11 = matrix_vector_mul(v.children[0], X1)
    Y_12 = matrix_vector_mul(v.children[1], X2)
    Y_21 = matrix_vector_mul(v.children[2], X1)
    Y_22 = matrix_vector_mul(v.children[3], X2)

    res = np.zeros((len(Y_11) + len(Y_12), len(X[0])))

    for i in range(len(Y_11)):
        for j in range(len(Y_11[i])):
            res[i][j] = Y_11[i][j] + Y_12[i][j]

    for i in range(len(Y_12)):
        for j in range(len(Y_12[i])):
            res[len(Y_11) + i][j] = Y_21[i][j] + Y_22[i][j]
    return res


if __name__ == "__main__":
    K = [2, 3, 4]
    T = []
    E = []
    global A
    for k in K:
        np.random.seed(100)

        A = generate_3d_grid_matrix(k)

        plot_binary_matrix(A, "macierz"+str(k))

        start_matrix = copy.deepcopy(A)

        n = len(A)
        m = 8
        global ax
        fig, ax = plt.subplots(figsize=(10.24, 10.24))
        # kompresja macierzy
        U, D, V = randomized_svd(A, n_components=n, random_state=0)
        v = create_tree(0, n, 0, n, 1, D[n - 1])
        vector = numpy.random.rand(n, m)

        show_matrices("h_macierz"+str(k), k)

        # mnozenie przez wektor
        start_time = time.time()
        res = matrix_vector_mul(v, vector)
        end_time = time.time()

        T.append(end_time-start_time)

        real_res = start_matrix @ vector
        E.append(np.sqrt(np.sum(np.power(res - real_res, 2))))

    plt.figure(figsize=(8, 6))
    plt.semilogy(K, T, marker='o')
    plt.title('Czas wykonania T(K)')
    plt.xlabel('K')
    plt.ylabel('Czas wykonania (s)')
    plt.grid(True)
    plt.xticks(K)
    plt.savefig('lab5/graphs/czas_wykonania.png')

    plt.figure(figsize=(8, 6))
    plt.bar(K, E, color='blue', alpha=0.7)
    plt.title(r'$||Ax - Hx||^2$ dla rozmiarów $2^{3k}$')
    plt.xlabel('k')
    plt.ylabel(r'$||Ax - Hx||^2$')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(K)
    plt.savefig('lab5/graphs/mse.png')
