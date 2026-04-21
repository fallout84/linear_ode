import numpy as np
from dataclasses import dataclass
from typing import Callable
import matplotlib.pyplot as plt


@dataclass
class BoundaryCondition:
    # -1 - дирихле, 1 - нейман
    left_type: int = 0.0
    right_type: int = 0.0
    left_val: float = 0.0
    right_val: float = 0.0


class ODE:
    def __init__(self, p: Callable, q: Callable, f: Callable, bc: BoundaryCondition) -> None:
        self.p = p
        self.q = q
        self.f = f
        self.bc = bc
        self.x_grid = None

    def generate_system(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        h = 1 / n
        self.x_grid = np.array([(i + 0.5) * h for i in range(n)])  # наш x
        a_matrix = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            xi = self.x_grid[i]  # берем из архива иксов
            pi, qi, fi = self.p(xi), self.q(xi), self.f(xi)

            w_prev = 1.0 - (pi * h) / 2.0  # i-1
            w_curr = -2.0 + (qi * h ** 2)  # i
            w_next = 1.0 + (pi * h) / 2.0  # i+1

            a_matrix[i, i] = w_curr
            b[i] = fi * h ** 2

            if i == 0:
                a_matrix[i, i] += self.bc.left_type * w_prev
                if self.bc.left_type == -1:  # Дирихле
                    b[i] -= w_prev * 2 * self.bc.left_val
                else:  # Нейман
                    b[i] -= w_prev * h * self.bc.left_val
            else:
                a_matrix[i, i - 1] = w_prev

            if i == n - 1:
                a_matrix[i, i] += self.bc.right_type * w_next
                if self.bc.right_type == -1:  # Дирихле
                    b[i] -= w_next * 2 * self.bc.right_val
                else:  # Нейман
                    b[i] += w_next * h * self.bc.right_val
            else:
                a_matrix[i, i + 1] = w_next

        return a_matrix, b


class LinearSolver:
    def __init__(self, tol: float = 1e-5, max_iter: int = 1000000) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.y: np.ndarray | None = None

    def solve(self, a_matrix: np.ndarray, b: np.ndarray) -> bool:
        pass

    def get_solution(self) -> np.ndarray | None:
        return self.y


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


class Legandre(ODE):  # kt;fylh
    def __init__(self, n: int):
        if n % 2 == 1:
            bc = BoundaryCondition(left_type=-1, right_type=-1, left_val=0, right_val=1)
        else:
            bc = BoundaryCondition(left_type=1, right_type=-1, left_val=0, right_val=1)

        super().__init__(
            p=lambda x: -2 * x / (1 - x ** 2),
            q=lambda x: n * (n + 1) / (1 - x ** 2),
            f=lambda x: 0,
            bc=bc)

# решение


plt.figure(figsize=(8, 6))
plt.grid(alpha=0.5)
for bnm in range(5):
    tak1 = Legandre(bnm)
    A1, b1 = tak1.generate_system(50)
    tak2 = Progonka()
    t = tak2.solve(A1, b1)
    y_res = tak2.get_solution()
    x_res = tak1.x_grid
    plt.plot(x_res, y_res, lw=0.67, ms=0, label=f"Analytical solution {bnm} ", zorder=3)


plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=14)
plt.show()
