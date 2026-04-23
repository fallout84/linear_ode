from problems.boundary_conditions import BoundaryCondition
from problems.lin_generator import ODE


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
