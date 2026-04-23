import numpy as np
from solver.Linear_solver import LinearSolver


class Jacobi(LinearSolver):
    def __init__(self, f: float = 0.01, tol: float = 1e-5, max_iter: int = 1000000):
        super().__init__(tol, max_iter)
        self.f = f

    def solve(self, a_matrix: np.ndarray, b: np.ndarray, fix: float = 0.01) -> bool:
        tau = np.diag(a_matrix)
        y = np.zeros(len(b))
        for _ in range(self.max_iter):
            r = b - a_matrix @ y
            if np.linalg.norm(r) < self.tol:
                self.y = y
                return True
            y += r * fix / tau
        return False
