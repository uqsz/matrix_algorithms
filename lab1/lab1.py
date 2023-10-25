# KOD PYTHON

import random
import matplotlib.pyplot as plt
import time
import numpy as np


def generate_matrix(n): # generates n-size matrix with numbers from range ( 10^(-8);1 )
    eps = np.finfo(float).eps
    return [[random.uniform(10 ** -8 + eps, 1.0 - eps) for _ in range(n)] for _ in range(n)]


def show_matrix(A): # print matrix
    for l in A:
        print(l)
    print("")
            

def add_matrix(A, B): # adds two matrices
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

def sub_matrix(A, B): # substracts two matrices
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


def normal_multiplication(A,B): # multiplicates two matrices with iteration method
    C = [[0 for _ in range(len(B))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            for k in range(len(A[0])):
                C[i][j] += A[i][k] * B[k][j]


def recursive_multiplication_binet(A, B): # multiplicates two matrices with Binet's method
    global cnt_m, cnt_a
    n = len(A)
    m = len(B)

    if m == 0:
        return [[]]

    l = len(B[0])

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

    A11 = [row[:m // 2] for row in A[:n // 2]]
    A12 = [row[m // 2:] for row in A[:n // 2]]
    A21 = [row[m // 2:] for row in A[n // 2:]]
    A22 = [row[:m // 2] for row in A[n // 2:]]

    A = [[A11, A12], [A22, A21]]

    B11 = [row[:l // 2] for row in B[:m // 2]]
    B12 = [row[l // 2:] for row in B[:m // 2]]
    B21 = [row[l // 2:] for row in B[m // 2:]]
    B22 = [row[:l // 2] for row in B[m // 2:]]

    B = [[B11, B12], [B22, B21]]

    C = [[0 for _ in range(len(A))] for _ in range(len(B[0]))]
    T = [[] for _ in range(len(B))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                T[k] = recursive_multiplication_binet(A[i][k], B[k][j])
            C[i][j] = add_matrix(T[0], T[1])

    res = []
    for i in range(2):
        for j in range(len(C[i][0])):
            res.append(C[i][0][j] + C[i][1][j])
    return res

def recursive_multiplication_strassen(A, B): # multiplicates two matrices with Strassen's method
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

    P1 = recursive_multiplication_strassen(add_matrix(A1, A4), add_matrix(B1, B4))
    P2 = recursive_multiplication_strassen(add_matrix(A3, A4), B1)
    P3 = recursive_multiplication_strassen(A1, sub_matrix(B2, B4))
    P4 = recursive_multiplication_strassen(A4, sub_matrix(B3, B1))
    P5 = recursive_multiplication_strassen(add_matrix(A1, A2), B4)
    P6 = recursive_multiplication_strassen(sub_matrix(A3, A1), add_matrix(B1, B2))
    P7 = recursive_multiplication_strassen(sub_matrix(A2, A4), add_matrix(B3, B4))

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


def test(n): # test function
    n_test = n
    tab_k = np.zeros(n_test)
    binet_time = []
    strassen_time = []
    binet_a = []
    strassen_a = []
    binet_m = []
    strassen_m = []
    binet_all = []
    strassen_all = []
    
    for k in range(2, 2 + n_test):
        global cnt_m, cnt_a
        
        tab_k[k-2] = k
        n = 2 ** k
        
        A = generate_matrix(n)
        B = generate_matrix(n)
    
        # Binet's test
        cnt_a = 0
        cnt_m = 0
        
        start = time.time()
        recursive_multiplication_binet(A, B)
        end = time.time()
        
        binet_time.append(end - start)
        binet_a.append(cnt_a)
        binet_m.append(cnt_m)
        binet_all.append(cnt_a + cnt_m)
        
        # Strassen's test
        cnt_a = 0
        cnt_m = 0
        
        start = time.time()
        recursive_multiplication_strassen(A, B)
        end = time.time()
        
        strassen_time.append(end - start)
        strassen_a.append(cnt_a)
        strassen_m.append(cnt_m)
        strassen_all.append(cnt_a + cnt_m)
    
    plt.clf()
    plt.plot(tab_k, binet_time, 'o-', label = "Binet")
    plt.plot(tab_k, strassen_time, 'o-', label = "Strassen")
    plt.legend()
    plt.title("Czas działania algorytmu od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Czas [s]")
    plt.savefig("wykres1.png")
    
    plt.clf()
    plt.plot(tab_k, binet_a, 'o-', label = "Binet")
    plt.plot(tab_k, strassen_a, 'o-', label = "Strassen")
    plt.legend()
    plt.title("Liczba dodawań od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Ilość dodawań (zlogarytmizowana)")
    plt.semilogy()
    plt.savefig("wykres2.png")
    
    plt.clf()
    plt.plot(tab_k, binet_m, 'o-', label = "Binet")
    plt.plot(tab_k, strassen_m, 'o-', label = "Strassen")
    plt.title("Liczba mnożeń od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Ilość mnożeń (zlogarytmizowana)")
    plt.legend()
    plt.semilogy()
    plt.savefig("wykres3.png")
    
    plt.clf()
    plt.plot(tab_k, binet_all, 'o-', label = "Binet")
    plt.plot(tab_k, strassen_all, 'o-', label = "Strassen")
    plt.title("Liczba działań arytmetycznych od rozmiaru macierzy (2^k x 2^k)")
    plt.xlabel("k")
    plt.ylabel("Liczba działań arytmetycznych (zlogarytmizowana)")
    plt.legend()
    plt.semilogy()
    plt.savefig("wykres4.png")
    
    
def test_matlab():
    global cnt_m, cnt_a
    
    cnt_a = 0
    cnt_m = 0
    
    
    A = [[0.9125387713535192, 0.45230209561129187],
         [0.07747586362850696, 0.1503258457223823]]

    B = [[0.6765535423510954, 0.943849754188833],
        [0.4181924832761693, 0.46715985459335185]]

    C_binet = recursive_multiplication_binet(A, B)
    C_strassen = recursive_multiplication_strassen(A, B)
    
    show_matrix(C_binet)
    show_matrix(C_strassen)
    
    
if __name__ == "__main__":
    test(8)
    # test_matlab()



# KOD MATLAB

# % Zmiana formatu na 'long' z większą precyzją
# format long;

# % Definicja pierwszej macierzy 3x3
# A = [0.9125387713535192, 0.45230209561129187;
#      0.07747586362850696, 0.1503258457223823];

# % Definicja drugiej macierzy 3x3
# B = [0.6765535423510954, 0.943849754188833;
#      0.4181924832761693, 0.46715985459335185];

# % Mnożenie macierzy A i B
# C = A * B;

# % Wyświetlenie wyniku
# disp('Macierz C (wynik mnożenia A i B):');
# disp(C);



    
    