import numpy as np
import random
from sklearn.utils.extmath import randomized_svd

random.seed(100)


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


def generate_matrix(n, proc):  # generates n-size matrix with numbers from range ( 10^(-8);1 )
    eps = np.finfo(float).eps
    matrix = np.zeros((n, n))
    idx = np.random.choice(n * n, int(n * n * proc / 100), replace=False)
    matrix.flat[idx] = random.uniform(10 ** -8 + eps, 1.0 - eps)
    return matrix


n = 2 ** 10
A = generate_matrix(n, 5)
r = np.linalg.matrix_rank(A)
# print(A)


def create_tree(t_min, t_max, s_min, s_max, r, eps):
    U, D, V = randomized_svd(
        A[t_min:t_max, s_min:s_max], n_components=r+1, random_state=0)
    if len(D) <= r or D[r] < eps:
        v = compress_matrix(t_min, t_max, s_min, s_max, U, D, V, r)
    else:
        v = Node((t_min, t_max, s_min, s_max))
        t_new_max = t_min + (t_max - t_min) // 2
        s_new_max = s_min + (s_max - s_min) // 2
        v.append(create_tree(t_min, t_new_max, s_min, s_new_max, r, eps))
        v.append(create_tree(t_min, t_new_max, s_new_max, s_max, r, eps))
        v.append(create_tree(t_new_max, t_max, s_min, s_new_max, r, eps))
        v.append(create_tree(t_new_max, t_max, s_new_max, s_max, r, eps))
        # TODO: połączyć macierze od dzieci w jedną macierz
        # v.children.sort(key=lambda x : (x.U.shape[0], x.U.shape[1]))
        print(v.children[0].U.shape, v.children[0].V.shape)
        print(v.children[1].U.shape, v.children[1].V.shape)
        print(v.children[2].U.shape, v.children[2].V.shape)
        print(v.children[3].U.shape, v.children[3].V.shape)
        print()

        v.U = np.vstack([np.hstack([v.children[0].U, v.children[1].U]), np.hstack(
            [v.children[2].U, v.children[3].U])])
        v.V = np.vstack([np.hstack([v.children[0].V, v.children[1].V]), np.hstack(
            [v.children[2].V, v.children[3].V])])
    return v


def compress_matrix(t_min, t_max, s_min, s_max, U, D, V, r):
    if np.all(A[t_min:t_max, s_min:s_max] == 0):
        v = Node((t_min, t_max, s_min, s_max))
        v.U = np.zeros((t_max - t_min, s_max - s_min))
        v.V = np.zeros((t_max - t_min, s_max - s_min))
        return v
    v = Node((t_min, t_max, s_min, s_max), r)
    v.sv = D[:r+1]
    v.U = U[:, :r + 1]
    v.V = np.diag(D[:r + 1]) @ V[:r + 1, :]
    # normalnie byłoby V[:r+1, :], ale tak jest w pseudokodzie
    # przez to w obliczaniu B mamy poprostu U @ V, a nie U @ np.diag(D) @ V
    return v


U, D, V = randomized_svd(A, n_components=n, random_state=0)
# r to w pseudokodzie parametr b, eps to parametr sigma
v = create_tree(0, n, 0, n, 1, D[n - 1])
B = v.U @ v.V
# print(A)
# print(B)
# print(U @ np.diag(D) @ V)
err = np.sqrt(np.sum(np.power(A - B, 2)))
print("Error:", err)
print(D)
# randomized_svd(A[0:4, 0:4], n_components=r+1, random_state=0)
