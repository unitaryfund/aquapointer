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
from scipy.stats import qmc

class Lattice:
    """ This is a class that contains information on 2D lattices"""

    def __init__(
        self,
        coords: ArrayLike,
        **kwargs
    ):
        self._coords = np.array(coords, dtype=float)
        try:
            self._min_spacing = kwargs['min_distance']
        except KeyError:
            self._min_spacing = emb.find_minimal_distance(self._coords)
        try:
            self._center = kwargs['center']
        except KeyError:
            self._center = np.mean(self._coords, axis=0)
        try:
            self._length_x = kwargs['length_x']
        except KeyError:
            self._length_x = emb.find_maximal_distance(self._coords[:,0])
        try:
            self._length_y = kwargs['length_y']
        except KeyError:
            self._length_y = emb.find_maximal_distance(self._coords[:,1])
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
            x_coords.append(i*spacing)
        
        y_coords = []
        for i in range(num_y):
            y_coords.append(i*spacing)

        coords = []
        for xc in x_coords:
            for yc in y_coords:
                coords.append([xc, yc])
       
        return cls(np.array(coords, dtype=float), min_spacing=spacing, length_x=length_x, length_y=length_y, type="rectangular")

    @classmethod
    def poisson_disk(cls, density, spacing, length, max_num):
        """
        Poisson disk sampling with variable radius.
        density: 2d array representing probability density
        spacing: tuple (min_radius, max_radius) representing the minimum and maximum exclusion radius
        length: tuple (length_x, length_y) representing the physical size of the 2d space
        max_num: maximum number of points to sample
        """
        
        min_radius, max_radius = spacing
        length_x, length_y = length
        scale_x, scale_y = np.array(length)/np.array(density.shape)

        def _index_from_position(pos):
            idx_x = int(pos[0]/scale_x)
            idx_y = int(pos[1]/scale_y)
            return (idx_x, idx_y)
        
        # convert probability density map to a radius map
        radius_density = -density #high probability = small radius and viceversa
        radius_density -= np.min(radius_density)
        radius_density *= (max_radius-min_radius)/np.max(radius_density)
        radius_density += min_radius
        # now radius_density is a 2d array with values between max_radius and min_radius

        coords = []
        queue = []
        probs = []
        num = 0

        # pick the first point randomly and initialize queue
        first_point = np.array((np.random.rand()*length_x, np.random.rand()*length_y))
        coords.append(first_point)
        queue.append(num)
        probs.append(density[_index_from_position(first_point)])
        num += 1

        # sample until max number is reached or points cannot be placed
        while len(queue) and (num<=max_num):
            i = np.random.choice(queue, p=np.array(probs)/np.sum(probs))
            ref_point = coords[i]
            ref_radius = radius_density[_index_from_position(ref_point)]
            placed = False

            # try placement 30 times (default hard-coded value)
            tries = 0
            while (tries < 30) and not placed:
                # pick a random point at random distance between [ref_radius, 2*ref_radius]
                r = (np.random.rand()+1)*ref_radius
                theta = 2*np.pi*np.random.rand()
                new_point = np.array((r*np.cos(theta), r*np.sin(theta))) + ref_point

                # burn a try if point falls outside space 
                if not (0 <= new_point[0] < length_x):
                    continue
                if not (0 <= new_point[1] < length_y):
                    continue

                new_radius = radius_density[_index_from_position(new_point)]

                # now check if new_point is too close to some other point based on new_radius
                too_close = False
                for other_point in coords:
                    d = np.linalg.norm(new_point-other_point)
                    if d < new_radius:
                        too_close = True
                        break

                if too_close:
                    tries +=1
                else:
                    coords.append(new_point)
                    queue.append(num)
                    probs.append(density[_index_from_position(new_point)])
                    num += 1
                    placed = True

            if not placed:
                # after 30 tries, no point could be placed around i. remove it from the active queue
                queue.remove(i)
                probs.remove(density[_index_from_position(coords[i])])
        
        return cls(np.array(coords, dtype=float), type="poisson_disk")

        











