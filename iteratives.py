import numpy as np
import numpy.linalg as la


def transform(A, b):
    """
    Ax = b -> x = Cx+d
    """
    C = A.copy()
    d = b.copy()
    for i in range(len(C)):
        d[i] /= C[i, i]
        C[i, :] /= -C[i, i]
        C[i, i] = 0
    return C, d


def richardson(A, b, tol, max_iter=100):
    """
    Richardson method, tau_k = ||A||
    returns: list of x, list of y
    """
    tau = la.norm(A)

    xs = []
    ys = []
    x = b / tau

    while len(xs) < max_iter:
        xs.append(x)
        err = la.norm(A @ x - b)
        ys.append(err)

        if err <= tol:
            break

        x = x - (A @ x - b)/tau

    return xs, ys


def jacobi(A, b, tol):
    """
    Jacobi method
    returns: list of x, list of y
    """
    xs = []
    ys = []
    C, d = transform(A, b)
    tau = la.norm(A)
    x = b / tau
    while True:
        xs.append(x)
        err = la.norm(A @ x - b)
        ys.append(err)
        if err < tol:
            break
        x = np.dot(C, x) + d
    return xs, ys


def seidel(A, b, tol):
    """
    Gauss-Seidel method
    returns: list of x, list of y
    """
    n = len(A)
    xs = []
    ys = []
    C, d = transform(A, b)
    tau = la.norm(A)
    x = b/tau
    while True:
        xs.append(x)
        err = np.linalg.norm(A @ x - b)
        ys.append(err)
        if err < tol:
            break
        for i in range(n):
            x[i] = C[i][:] @ x + d[i]
    return xs, ys
