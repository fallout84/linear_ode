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
