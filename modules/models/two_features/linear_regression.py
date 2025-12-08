import numpy as np

def least_squares(
    X: np.ndarray[np.double, np.double], 
    y: np.ndarray[np.double]
) -> np.ndarray[np.double]:
    n = len(y)
    c = np.ones(n).reshape(-1, 1)
    X = np.hstack((c, X))

    Xt = X.transpose()

    XtX = Xt @ X
    Xty = Xt @ y

    return np.linalg.solve(XtX, Xty)

