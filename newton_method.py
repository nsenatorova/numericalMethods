import numpy as np
import sympy as sp
from sympy.abc import x, y


def solve_scalar(f: sp.Expr, x0, tol):
    """
    solve scalar equation y=f(x) starting with x0
    derivative can be obtained as f.diff()
    evaluate: float(f.subs(x, x0))
    """

    xs = []
    ys = []
    x1 = x0
    i = 1
    max_iterations = 500
    xs.append(x0)
    ys.append(float(f.evalf(subs={x: x0})))
    while i < max_iterations:
        x_prev = x1
        x1 = x1 - float(f.evalf(subs={x: x1})) / float(f.diff().evalf(subs={x: x1}))  # x(n+1) = x(n) - f(x(n))/f'(x(n))
        xs.append(x1)
        ys.append(float(f.evalf(subs={x: x1})))
        err = abs(x1 - x_prev)
        if err < tol:
            break
        i += 1
    return xs, ys


def solve_plane(f: sp.Matrix, x0, y0, tol):
    """
    solve SAE {f1(x,y) = 0, f2(x,y) = 0} starting with (x0,y0)
    jacobian can be obtained as f.jacobian([x, y])
    """
    xs = []
    ys = []
    zs = []
    xs.append(x0)
    ys.append(y0)
    X1 = np.array([x0, y0])
    max_iterations = 500
    jacobian = sp.Matrix(f.jacobian([x, y]))
    jacobian_inv = jacobian.inv()
    for i in range(max_iterations):
        X_prev = X1
        inv_x = jacobian_inv.evalf(subs={x: X1[0], y: X1[1]})
        func_x = f.evalf(subs={x: X1[0], y: X1[1]})
        mul = np.dot(inv_x, func_x)
        X1 = np.array([X1[0] - mul[0], X1[1] - mul[1]], dtype=np.float64)
        X1 = np.array([X1[0][0], X1[1][0]])
        xs.append(X1[0])
        ys.append(X1[1])
        err = np.linalg.norm(abs(np.array(f.evalf(subs={x: X1[0], y: X1[1]}), dtype=np.float64) - np.array(
            f.evalf(subs={x: X_prev[0], y: X_prev[1]}), dtype=np.float64)))
        zs.append(err)
        if err <= tol:
            return xs, ys, zs
    return xs, ys, zs
