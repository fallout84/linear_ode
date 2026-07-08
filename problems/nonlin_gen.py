from problems.lin_generator import ODE
from problems.lin_generator import BoundaryCondition
import numpy as np
from solver.Linear import Progonka


class ODENewton:
    def __init__(self, f1: callable, bc: BoundaryCondition) -> None:
        self.f1 = f1    # функция из ур. y''=f1
        self.y0 = 0  # решение
        self.bc = bc
        self.x_grid = None     # cетка по x создается в def solve
        self.delta = 1e-7      # параметр для численного взятия производной
        self.err = []           # архив ошибок
        self.iter_needed = 1000  # счетчик итераций
        self.max_iter = 1000   # макс кол-во итераций  def solve
        self.gamma = 1e-5       # точность подбора решения

    def set_linearisation_point(self, y: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_dy = (self.f1(self.x_grid, y + self.delta, dy)
                 - self.f1(self.x_grid, y - self.delta, dy)) / (2 * self.delta)
        f_base = self.f1(self.x_grid, y, dy)
        df_ddy = (self.f1(self.x_grid, y, dy + self.delta) -
                  self.f1(self.x_grid, y, dy - self.delta)) / (2 * self.delta)
        return -df_ddy, -df_dy, f_base - df_dy * y - df_ddy * dy
    # я точно помню что уже писал взятие производной, но не помню где

    def proizvodnya(self, y: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        dy = np.zeros(n)
        d2y = np.zeros(n)
        if self.bc.left_type == -1:
            leftot = 2 * self.bc.left_val - y[0]
        else:
            leftot = y[0] - h * self.bc.left_val
        if self.bc.right_type == -1:
            rightot = 2 * self.bc.right_val - y[-1]
        else:
            rightot = y[-1] + h * self.bc.right_val
        v_ext = np.concatenate(([leftot], y, [rightot]))
        for i in range(n):
            v_prev = v_ext[i]
            v_curr = v_ext[i + 1]
            v_next = v_ext[i + 2]
            dy[i] = (v_next - v_prev) / (2 * h)
            d2y[i] = (v_next - 2 * v_curr + v_prev) / (h ** 2)
        return dy, d2y

    def solve(self, n: int) -> bool:
        self.err = []
        h = 1 / n
        self.x_grid = np.array([(i + 0.5) * h for i in range(n)])
        y = (self.bc.right_val-self.bc.left_val)*self.x_grid + self.bc.left_val
        dy, d2y = self.proizvodnya(y, h)

        for _ in range(self.max_iter):
            p_arr, q_arr, f_arr = self.set_linearisation_point(y, dy)
            p = lambda xi: p_arr[int(round(xi / h - 0.5))]
            q = lambda xi: q_arr[int(round(xi / h - 0.5))]
            f = lambda xi: f_arr[int(round(xi / h - 0.5))]

            # это ОДЕ
            tak1: ODE = ODE(p, q, f, self.bc)

            a1, b1 = tak1.generate_system(n)
            solver1 = Progonka()
            solver1.solve(a1, b1)
            y1 = solver1.get_solution()
            dy1, d2y1 = self.proizvodnya(y1, h)
            r = np.max(np.abs(d2y1 - self.f1(self.x_grid, y1, dy1)))
            self.err.append(r)
            if r > 10000:
                print("не сошлось")
                return True
            if r <= self.gamma:
                self.iter_needed = _
                self.y0 = y1
                return True
            y = y1
            dy = dy1
        return False

    def get_solution(self):
        return self.y0
