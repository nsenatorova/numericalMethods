import numpy as np
import numpy.linalg as la


def solve_upper_triang(A, b):
    """
    solves Ax=b for U-triangular A
    """
    A = np.array(A, ndmin=2)
    b = np.reshape(b, (-1, 1))
    n = len(b)

    Ab = np.concatenate((A, b), axis=1)

    for i in range(n)[::-1]:
        Ab[i, :] /= Ab[i, i]
        for j in range(i):
            Ab[j, :] -= Ab[i, :] * Ab[j, i]

    A, b = np.split(Ab, [n], axis=1)
    return b.reshape(-1)


def solve_lower_triang(A, b):
    """
    solves Ax=b for L-triangular A
    """
    A = np.array(A, ndmin=2)
    b = np.reshape(b, (-1, 1))
    n = len(b)
    Ab = np.concatenate((A, b), axis=1)
    for i in range(n):
        Ab[i, :] /= Ab[i, i]
        for j in range(i + 1, n):
            Ab[j, :] -= Ab[i, :] * Ab[j, i]
    A, b = np.split(Ab, [n], axis=1)
    return b.reshape(-1)


def lu(A):
    """
    LU decomposition
    """
    LU = np.matrix(np.zeros([A.shape[0], A.shape[1]]))
    n = A.shape[0]
    for k in range(n):
        for j in range(k, n):
            LU[k, j] = A[k, j] - LU[k, :k] * LU[:k, j]
        for i in range(k + 1, n):
            LU[i, k] = (A[i, k] - LU[i, : k] * LU[: k, k]) / LU[k, k]
    L = LU.copy()
    U = LU.copy()
    for i in range(L.shape[0]):
        L[i, i] = 1
        L[i, i + 1:] = 0
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return L, U


def qr(A):
    """
    QR decomposition
    """
    q = np.zeros_like(A)
    r = np.zeros_like(A)
    for i in range(0, A.shape[0]):
        q[:, i] = A[:, i]
        for j in range(0, i):
            r[j][i] = np.dot(q[:, j].T, A[:, i])
            q[:, i] = q[:, i] - np.dot(r[j][i], q[:, j])
        r[i][i] = np.linalg.norm(q[:, i])
        q[:, i] = q[:, i] / r[i][i]
    return q, r


def solve_lu(A, b):
    """
    solve Ax=b using LU decomposition
    """
    L, U = lu(A)
    b1 = solve_lower_triang(L, b)
    return solve_upper_triang(U, b1)


def solve_qr(A, b):
    """
    solve Ax=b using QR decomposition
    """
    Q, R = qr(A)
    b = np.reshape(b, (-1, 1))
    return solve_upper_triang(R, Q.T @ b)
