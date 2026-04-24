import numpy as np


class LinearSolver:
    def __init__(self, tol: float = 1e-5, max_iter: int = 1000000) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.y: np.ndarray | None = None

    def solve(self, a_matrix: np.ndarray, b: np.ndarray) -> bool:
        pass

    def get_solution(self) -> np.ndarray | None:
        return self.y
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
class Progonka(LinearSolver):
    def solve(self, a_matrix: np.ndarray, b: np.ndarray) -> bool:
        n = len(b)
        alpha = np.zeros(n)
        beta = np.zeros(n)

        alpha[0] = -a_matrix[0, 1] / a_matrix[0, 0]
        beta[0] = b[0] / a_matrix[0, 0]

        for i in range(1, n):
            d = a_matrix[i, i] + a_matrix[i, i - 1] * alpha[i - 1]
            if i < n - 1:
                alpha[i] = -a_matrix[i, i + 1] / d
            beta[i] = (b[i] - a_matrix[i, i - 1] * beta[i - 1]) / d

        self.y = np.zeros(n)
        self.y[-1] = beta[-1]
        for i in range(n - 2, -1, -1):
            self.y[i] = alpha[i] * self.y[i + 1] + beta[i]

        return True
