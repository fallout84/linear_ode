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
