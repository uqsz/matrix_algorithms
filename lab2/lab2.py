import random
import matplotlib.pyplot as plt
import time
import numpy as np


def generate_matrix(n):  # generates n-size matrix with numbers from range ( 10^(-8);1 )
    eps = np.finfo(float).eps
    # return [[random.uniform(10 ** -8 + eps, 1.0 - eps) for _ in range(n)]
    #         for _ in range(n)]
    return [[random.randint(0, 100) for _ in range(n)] for _ in range(n)]


def generate_identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def show_matrix(A):  # print matrix
    for l in A:
        print(l)
    print("")


def show_matrix_round(A):
    for row in A:
        formatted_row = ["{:.6f}".format(x) for x in row]
        print(formatted_row)
    print("")


def add_matrix(A, B):  # adds two matrices
    global cnt_a

    if (A == [[]]):
        return B
    if (B == [[]]):
        return A

    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
            # cnt_a += 1
    return C


def sub_matrix(A, B):  # substracts two matrices
    global cnt_a

    if (A == [[]]):
        C = [[0 for _ in range(len(B[0]))] for _ in range(len(B))]
        for i in range(len(B)):
            for j in range(len(B[0])):
                C[i][j] = -B[i][j]
                # cnt_a += 1
        return C
    if (B == [[]]):
        return A

    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] - B[i][j]
            # cnt_a += 1
    return C


def neg_matrix(A):
    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] = -A[i][j]
    return A

# multiplicates two matrices with Strassen's method


def mul_matrix(A, B):
    # global cnt_m, cnt_a
    n = len(A)
    m = len(B)
    l = len(B[0])

    if m == 0:
        return [[]]

    if l == 1:
        C = [[0] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                C[i][0] += A[i][j] * B[j][0]
                # cnt_m += 1
                # cnt_a += 1
        return C

    if n == 1:
        C = [[0 for _ in range(l)]]
        for i in range(m):
            for j in range(l):
                C[0][j] += A[0][i] * B[i][j]
                # cnt_m += 1
                # cnt_a += 1
        return C

    A1 = [row[:m // 2] for row in A[:n // 2]]
    A2 = [row[m // 2:] for row in A[:n // 2]]
    A4 = [row[m // 2:] for row in A[n // 2:]]
    A3 = [row[:m // 2] for row in A[n // 2:]]

    B1 = [row[:l // 2] for row in B[:m // 2]]
    B2 = [row[l // 2:] for row in B[:m // 2]]
    B4 = [row[l // 2:] for row in B[m // 2:]]
    B3 = [row[:l // 2] for row in B[m // 2:]]

    P1 = mul_matrix(add_matrix(A1, A4), add_matrix(B1, B4))
    P2 = mul_matrix(add_matrix(A3, A4), B1)
    P3 = mul_matrix(A1, sub_matrix(B2, B4))
    P4 = mul_matrix(A4, sub_matrix(B3, B1))
    P5 = mul_matrix(add_matrix(A1, A2), B4)
    P6 = mul_matrix(sub_matrix(A3, A1), add_matrix(B1, B2))
    P7 = mul_matrix(sub_matrix(A2, A4), add_matrix(B3, B4))

    C = [[0 for _ in range(2)] for _ in range(2)]
    C[0][0] = add_matrix(sub_matrix(add_matrix(P1, P4), P5), P7)
    C[0][1] = add_matrix(P3, P5)
    C[1][0] = add_matrix(P2, P4)
    C[1][1] = add_matrix(add_matrix(sub_matrix(P1, P2), P3), P6)

    res = []
    for i in range(2):
        for j in range(len(C[i][0])):
            res.append(C[i][0][j] + C[i][1][j])
    return res


def gauss_inverse(A):
    n = len(A)
    I = generate_identity(n)
    A = [A[i] + I[i] for i in range(n)]
    for i in range(n):
        temp = A[i][i]
        for j in range(i, 2*n):
            A[i][j] /= temp
        for j in range(i+1, n):
            temp = A[j][i]
            for k in range(i, 2*n):
                A[j][k] -= temp*A[i][k]
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            temp = A[j][i]
            for k in range(2*n-1, i-1, -1):
                A[j][k] -= temp*A[i][k]
    return [A[i][n:] for i in range(n)]


def inverse(A):
    n = len(A)
    if n == 2:
        return gauss_inverse(A)

    A11 = [row[:n // 2] for row in A[:n // 2]]
    A12 = [row[n // 2:] for row in A[:n // 2]]
    A21 = [row[:n // 2] for row in A[n // 2:]]
    A22 = [row[n // 2:] for row in A[n // 2:]]

    A11_inv = inverse(A11)
    S22 = sub_matrix(A22, mul_matrix(mul_matrix(A21, A11_inv), A12))
    S22_inv = inverse(S22)

    C1 = mul_matrix(A11_inv, A12)
    C2 = mul_matrix(mul_matrix(S22_inv, A21), A11_inv)

    B = [[0 for _ in range(2)] for _ in range(2)]

    B[0][0] = add_matrix(A11_inv, mul_matrix(C1, C2))
    B[0][1] = neg_matrix(mul_matrix(C1, S22_inv))
    B[1][0] = neg_matrix(C2)
    B[1][1] = S22_inv

    res = []
    for i in range(2):
        for j in range(len(B[i][0])):
            res.append(B[i][0][j] + B[i][1][j])
    return res


def error(A, B):
    return round((np.linalg.norm(A - B, 'fro') / np.linalg.norm(B, 'fro'))*100, 2)


A = generate_matrix(8)
# A = [[2, 1, 4, 2],
#      [4, 4, 3, 0],
#      [0, 1, 0, 1],
#      [1, 1, 3, 3]]
# A = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 2, 3, 4], [1, 2, 5, 1]]

C = inverse(A)
D = np.linalg.inv(A)

# show_matrix_round(B)
show_matrix_round(C)
show_matrix_round(D)

print(error(C, D), "%")
# print(error(B, D),"%")
