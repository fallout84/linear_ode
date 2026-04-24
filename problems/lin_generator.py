import numpy as np
from typing import Callable
from dataclasses import dataclass


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


class Chebyshev(ODE):
    def __init__(self, n: int):
        if n % 2 != 0:
            bc = BoundaryCondition(-1, -1, 0, 1)
        else:
            bc = BoundaryCondition(1, -1, 0, 1)
        super().__init__(
            p=lambda x: -x/(1-x**2),
            q=lambda x: n**2/(1-x**2),
            f=lambda x: 0,
            bc=bc)


class Legandre(ODE):  
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
