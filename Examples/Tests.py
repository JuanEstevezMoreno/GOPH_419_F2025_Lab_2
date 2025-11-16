import numpy as np
from scipy.interpolate import UnivariateSpline
from linalg_interp import gauss_iter_solve, spline_function

def test_gauss_solver_single_rhs():
    A = np.array([[4,1],[2,3]], float)
    b = np.array([1,2],float)
    x = gauss_iter_solve(A, b)
    expected = np.linalg.solve(A, b)
    print("GS single RHS:", np.allclose(x.flatten(), expected))

def test_gauss_solver_inverse():
    A = np.array([[4,1],[2,3]], float)
    I = np.eye(2)
    X = gauss_iter_solve(A, I)
    print("GS inverse:", np.allclose(A @ X, I))

def test_spline_exact_recovery():
    xd = np.linspace(0, 10, 6)
    yd = 3*xd + 2
    f = spline_function(xd, yd, order=1)
    ytest = f(xd)
    print ("Linear spline exact:", np.allclose(yd, ytest))

def test_spline_vs_scipy():
    xd = np.linspace(0,5,20)
    yd = np.exp(xd)

    f = spline_function(xd, yd, order=3)
    g = UnivariateSpline (xd, yd, k=3, s=0, ext='raise')

    xs = np.linspace(0, 5, 200)
    print("Cubic spline vs scipy:", np.allclose(f(xs), g(xs)))

if __name__ == "__main__":
    test_gauss_solver_single_rhs()
    test_gauss_solver_inverse()
    test_spline_exact_recovery()
    test_spline_vs_scipy()