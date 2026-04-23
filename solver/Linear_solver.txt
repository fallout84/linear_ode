class LinearSolver:
    def __init__(self, tol: float = 1e-5, max_iter: int = 1000000) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.y: np.ndarray | None = None

    def solve(self, a_matrix: np.ndarray, b: np.ndarray) -> bool:
        pass

    def get_solution(self) -> np.ndarray | None:
        return self.y

