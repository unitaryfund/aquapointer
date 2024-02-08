import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Callable
from typing import Union
from typing_extensions import Self
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numbers
import math
import Lp_norm as lpn
import embedding as emb
from itertools import combinations

class Lattice:
    """ This is a class that contains information on 2D lattices"""

    def __init__(
        self,
        coords: ArrayLike,
        **kwargs
    ):
        self._coords = np.array(coords)
        try:
            self._min_spacing = kwargs['min_distance']
        except KeyError:
            self._min_spacing = emb.find_minimal_distance(coords)
        try:
            self._length_x = kwargs['length_x']
        except KeyError:
            self._length_x = emb.find_maximal_distance(coords[:,0])
        try:
            self._length_y = kwargs['length_y']
        except KeyError:
            self._length_y = emb.find_maximal_distance(coords[:,1])
        try:
            self._lattice_type = kwargs['lattice_type']
        except KeyError:
            self._lattice_type = "custom"

    @classmethod
    def rectangular(cls, num_x: int, num_y: int, spacing: numbers.Number):
        length_x = (num_x-1)*spacing
        length_y = (num_y-1)*spacing
        
        x_coords = []
        for i in range(num_x):
            x_coords.append(i*spacing-(length_x/2))
        
        y_coords = []
        for i in range(num_y):
            y_coords.append(i*spacing-(length_y/2))

        coords = []
        for xc in x_coords:
            for yc in y_coords:
                coords.append([xc, yc])
        
        return cls(np.array(coords), min_spacing=spacing, length_x=length_x, length_y=length_y, type="rectangular")