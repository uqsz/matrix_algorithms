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
            cnt_a += 1
    return C


def sub_matrix(A, B):  # substracts two matrices
    global cnt_a

    if (A == [[]]):
        C = [[0 for _ in range(len(B[0]))] for _ in range(len(B))]
        for i in range(len(B)):
            for j in range(len(B[0])):
                C[i][j] = -B[i][j]
                cnt_a += 1
        return C
    if (B == [[]]):
        return A

    C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] - B[i][j]
            cnt_a += 1
    return C


def neg_matrix(A):
    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] = -A[i][j]
    return A

# multiplicates two matrices with Strassen's method


def mul_matrix(A, B):
    global cnt_m, cnt_a
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
                cnt_m += 1
                cnt_a += 1
        return C

    if n == 1:
        C = [[0 for _ in range(l)]]
        for i in range(m):
            for j in range(l):
                C[0][j] += A[0][i] * B[i][j]
                cnt_m += 1
                cnt_a += 1
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

def doolittle_LU(A):
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        L[i][i] = 1

    for i in range(n):
        for j in range(i, n):
            tmp = 0
            for k in range(i):
                tmp += L[i][k] * U[k][j]
            U[i][j] = A[i][j] - tmp

        for j in range(i + 1, n):
            tmp = 0
            for k in range(i):
                tmp += L[j][k] * U[k][i]
            L[j][i] = (A[j][i] - tmp) / U[i][i]

    return [L, U]

def LU(A):
    n = len(A)
    if n == 2:
        return doolittle_LU(A)

    A11 = [row[:n // 2] for row in A[:n // 2]]
    A12 = [row[n // 2:] for row in A[:n // 2]]
    A21 = [row[:n // 2] for row in A[n // 2:]]
    A22 = [row[n // 2:] for row in A[n // 2:]]

    L11, U11 = LU(A11)
    U11_inv = inverse(U11)
    L21 = mul_matrix(A21, U11_inv)
    L11_inv = inverse(L11)
    U12 = mul_matrix(L11_inv, A12)
    S = sub_matrix(A22, mul_matrix(mul_matrix(mul_matrix(A21, U11_inv), L11_inv), A12))
    L22, U22 = LU(S)

    U = [[U11, U12], [[[0 for _ in range(n - len(U22))] for _ in range(n - len(U22))], U22]]
    L = [[L11, [[0 for _ in range(n - len(L11))] for _ in range(n - len(L11))]], [L21, L22]]

    U_res = []
    for i in range(2):
        for j in range(len(U[i][0])):
            U_res.append(U[i][0][j] + U[i][1][j])

    L_res = []
    for i in range(2):
        for j in range(len(L[i][0])):
            L_res.append(L[i][0][j] + L[i][1][j])

    return [L_res, U_res]

def determinant(A):
    if len(A) != len(A[0]):
        print("Error")
        return 0
    global cnt_m
    L, U = LU(A)
    res = 1
    for i in range(len(A)):
        res *= L[i][i] * U[i][i]
        cnt_m += 2
    return res

def error(A, B):
    return (np.linalg.norm(A - B, 'fro') / np.linalg.norm(B, 'fro'))*100


def test(n):  # test function
    n_test = n
    tab_k = np.zeros(n_test)
    inverse_time = []
    LU_time = []
    determinant_time = []
    inverse_a = []
    LU_a = []
    determinant_a = []
    inverse_m = []
    LU_m = []
    determinant_m = []
    inverse_all = []
    LU_all = []
    determinant_all = []

    for k in range(2, 2 + n_test):
        global cnt_m, cnt_a

        tab_k[k - 2] = k
        n = 2 ** k

        A = generate_matrix(n)

        # inverse matrix test
        cnt_a = 0
        cnt_m = 0

        start = time.time()
        inverse(A)
        end = time.time()

        inverse_time.append(end - start)
        inverse_a.append(cnt_a)
        inverse_m.append(cnt_m)
        inverse_all.append(cnt_a + cnt_m)

        # LU test
        cnt_a = 0
        cnt_m = 0

        start = time.time()
        LU(A)
        end = time.time()

        LU_time.append(end - start)
        LU_a.append(cnt_a)
        LU_m.append(cnt_m)
        LU_all.append(cnt_a + cnt_m)

        # determinant test
        cnt_a = 0
        cnt_m = 0

        start = time.time()
        determinant(A)
        end = time.time()

        determinant_time.append(end - start)
        determinant_a.append(cnt_a)
        determinant_m.append(cnt_m)
        determinant_all.append(cnt_a + cnt_m)

    plt.clf()
    plt.plot(tab_k, inverse_time, 'o-')
    plt.title("Czas obliczania odwrotności od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Czas [s]")
    plt.savefig("inverse_time.png")

    plt.clf()
    plt.plot(tab_k, inverse_a, 'o-', label="dodawanie")
    plt.plot(tab_k, inverse_m, 'o-', label="mnożenie")
    plt.plot(tab_k, inverse_all, 'o-', label="łącznie")
    plt.legend()
    plt.title("Liczba operacji od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Ilość operacji (zlogarytmizowana)")
    plt.semilogy()
    plt.savefig("inverse_operations.png")

    plt.clf()
    plt.plot(tab_k, LU_time, 'o-')
    plt.title("Czas obliczania faktoryzacji LU od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Czas [s]")
    plt.savefig("LU_time.png")

    plt.clf()
    plt.plot(tab_k, LU_a, 'o-', label="dodawanie")
    plt.plot(tab_k, LU_m, 'o-', label="mnożenie")
    plt.plot(tab_k, LU_all, 'o-', label="łącznie")
    plt.legend()
    plt.title("Liczba operacji od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Ilość operacji (zlogarytmizowana)")
    plt.semilogy()
    plt.savefig("LU_operations.png")

    plt.clf()
    plt.plot(tab_k, determinant_time, 'o-')
    plt.title("Czas obliczania wyznacznika od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Czas [s]")
    plt.savefig("determinant_time.png")

    plt.clf()
    plt.plot(tab_k, determinant_a, 'o-', label="dodawanie")
    plt.plot(tab_k, determinant_m, 'o-', label="mnożenie")
    plt.plot(tab_k, determinant_all, 'o-', label="łącznie")
    plt.legend()
    plt.title("Liczba operacji od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Ilość operacji (zlogarytmizowana)")
    plt.semilogy()
    plt.savefig("determinant_operations.png")

if __name__ == "__main__":
    test(8)
