import numpy as np
import random
from sklearn.utils.extmath import randomized_svd
import time
import matplotlib.pyplot as plt
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
    return v


def decompress(B, v):
    if v.children == []:
        B[v.size[0]: v.size[1], v.size[2]: v.size[3]] = v.U @ v.V
        return B
    for child in v.children:
        B = decompress(B, child)
    return B


random.seed(100)
n = 2 ** 10
selfvalues = []
times = []
errors = []


for l, proc in enumerate([1, 2, 5, 10, 20]):
    A = generate_matrix(n, proc)
    B = np.zeros((n, n))
    U, D, V = randomized_svd(A, n_components=n, random_state=0)

    selfvalues.append(D)

    start_time = time.time()
    v = create_tree(0, n, 0, n, 1, D[n-1])
    end_time = time.time()

    B = decompress(B, v)

    times.append(end_time-start_time)
    errors.append(np.sqrt(np.sum(np.power(A - B, 2))))
    print(f"proc = {proc}, b = {1}, error = {errors[-1]}")
    # for j, b in enumerate([1, 4]):
    #     for k, i in enumerate([2, n//2 - 1, n - 1]):

    #         v = create_tree(0, n, 0, n, b, D[i])

    #         B = decompress(B, v)
    #         err = np.sqrt(np.sum(np.power(A - B, 2)))

    #         print(f"proc = {proc}, b = {b}, i = {i}, error = {err}")


fig, ax = plt.subplots()
d = [1, 2, 5, 10, 20]
for i, values in enumerate(selfvalues):
    ax.plot(values, label=f'{d[i]}% wartości niezerowych')

ax.set_xlabel('Indeks')
ax.set_ylabel('Wartość')
ax.set_title('Wartości osobliwe macierzy')
ax.legend()

plt.savefig('lab3/wartosci_wlasne.png')

d = ["1%", "2%", "5%", "10%", "20%"]

fig, ax = plt.subplots()
ax.bar(d, times)

ax.set_xlabel('Procent wartości niezerowych')
ax.set_ylabel('Czas (s)')
ax.set_title('Czas kompresji')
ax.legend()

plt.savefig('lab3/czas_kompresji.png')

fig, ax = plt.subplots()
ax.bar(d, errors)

ax.set_xlabel('Procent wartości niezerowych')
ax.set_ylabel('Wartosc')
ax.set_title('MSE')
ax.legend()

plt.savefig('lab3/mse.png')
