import numpy as np
from numpy.typing import ArrayLike
import numbers
import math


def find_minimal_distance(pos: ArrayLike) -> numbers.Number:
    min_spacing = math.inf
    for i in range(len(pos)):
        pi = np.array(pos[i])
        for j in range(i+1, len(pos)):
            pj = np.array(pos[j])
            min_spacing = min(min_spacing, np.linalg.norm(pi-pj))
    return min_spacing