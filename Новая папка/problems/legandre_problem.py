@dataclass
class BoundaryCondition:
    # -1 - дирихле, 1 - нейман
    left_type: int = 0.0
    right_type: int = 0.0
    left_val: float = 0.0
    right_val: float = 0.0
